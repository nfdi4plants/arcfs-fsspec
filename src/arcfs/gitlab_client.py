from __future__ import annotations

import asyncio
import json
import warnings
from collections.abc import Awaitable, Callable
from typing import Any, Optional
from urllib.parse import quote

import aiohttp

from .errors import RefNotFound
from .transactions import commit_lfs_transaction
from .utils import calculate_sha256


def _message_from_text(text: str) -> str:
    """Return what GitLab said went wrong, given a body already read as text."""
    try:
        body = json.loads(text)
    except Exception:  # noqa: BLE001 - a body that will not parse simply said nothing
        return ""
    if not isinstance(body, dict):
        return ""
    return str(body.get("message") or body.get("error") or "")


async def _gitlab_message(response) -> str:
    """
    Return what GitLab said went wrong, or ``""`` if it did not say.

    GitLab reports a refused request under ``message`` and a malformed one under ``error``,
    so reading only one of them loses half the reasons a call can fail. Anything else, an
    HTML error page from a proxy or a truncated body included, is treated as having said
    nothing: this runs only to explain a failure that is already certain, so anything it
    raised would replace the error it was called to describe with a worse one.
    """
    try:
        body = await response.json()
    except Exception:  # noqa: BLE001 - see above
        return ""
    if not isinstance(body, dict):
        return ""
    return str(body.get("message") or body.get("error") or "")


class GitLabClient:
    """
    Async GitLab REST and LFS client used by the fsspec filesystem.

    The client owns an aiohttp session, builds GitLab API requests, normalizes
    project metadata for the filesystem, and exposes helpers for listings,
    file streaming, commits, merge requests, and Git LFS uploads.
    """

    def __init__(self, base_url: str, token: Optional[str]):
        """
        Create a GitLab API client.

        Args:
            base_url: Base GitLab instance URL, for example
                ``"https://gitlab.example.com"``.
            token: Private token used for GitLab API authentication, or
                ``None`` for unauthenticated requests.
        """
        self.base_url = base_url.rstrip("/")
        self.token = token

        self._session: aiohttp.ClientSession | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def _ensure(self) -> aiohttp.ClientSession:
        """
        Ensure we have a ClientSession bound to the current running event loop.

        If the loop changes (common when mixing sync wrappers + asyncio.run),
        the old session must be closed and recreated, otherwise aiohttp will throw.

        Returns:
            The active ``aiohttp.ClientSession`` for the current event loop.
        """
        loop = asyncio.get_running_loop()

        if self._loop is not None and self._loop is not loop:
            await self.close()

        if self._session is None or self._session.closed:
            headers: dict[str, str] = {}
            if self.token:
                headers["PRIVATE-TOKEN"] = self.token
            self._session = aiohttp.ClientSession(headers=headers)
            self._loop = loop

        return self._session

    async def close(self) -> None:
        """
        Close the active HTTP session and clear loop-bound client state.

        Returns:
            None.
        """
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        self._loop = None

    # ------------------------------------------------------------------
    # Private helpers: URLs / params / header parsing
    # ------------------------------------------------------------------
    def _projects_url(self) -> str:
        """Build the GitLab projects collection API URL."""
        return f"{self.base_url}/api/v4/projects"

    def _project_url(self, repo_id: int) -> str:
        """Build the GitLab project metadata API URL for one project id."""
        return f"{self.base_url}/api/v4/projects/{repo_id}"

    def _project_by_path_url(self, path_with_namespace: str) -> str:
        """Build the GitLab project lookup API URL for a namespaced project path."""
        return f"{self.base_url}/api/v4/projects/{quote(path_with_namespace, safe='')}"

    def _project_tree_url(self, repo_id: int) -> str:
        """Build the GitLab repository tree API URL for one project id."""
        return f"{self.base_url}/api/v4/projects/{repo_id}/repository/tree"

    def _repository_file_url(self, repo_id: int, path: str) -> str:
        """Build the GitLab repository file API URL for one repository path."""
        return (
            f"{self.base_url}/api/v4/projects/{repo_id}/repository/files/"
            f"{quote(path, safe='')}"
        )

    def _repository_file_raw_url(self, repo_id: int, path: str) -> str:
        """Build the raw-content URL for one GitLab repository file."""
        return f"{self._repository_file_url(repo_id, path)}/raw"

    def _branches_url(self, repo_id: int) -> str:
        """Build the GitLab branches API URL for one project id."""
        return f"{self.base_url}/api/v4/projects/{repo_id}/repository/branches"

    def _commits_url(self, repo_id: int) -> str:
        """Build the GitLab commits API URL for one project id."""
        return f"{self.base_url}/api/v4/projects/{repo_id}/repository/commits"

    def _merge_requests_url(self, repo_id: int) -> str:
        """Build the GitLab merge requests API URL for one project id."""
        return f"{self.base_url}/api/v4/projects/{repo_id}/merge_requests"

    def _lfs_batch_url(self, namespace: str) -> str:
        """Build the Git LFS batch URL for a project namespace."""
        return f"{self.base_url}/{namespace}.git/info/lfs/objects/batch"

    def _extract_next_link(self, link_header: str | None) -> str | None:
        """
        Return the URL for rel="next" from a GitLab Link header.

        Args:
            link_header: Raw ``Link`` response header, or ``None`` when the
                response did not include one.

        Returns:
            The next-page URL as ``str`` when present, otherwise ``None``.
        """
        if not link_header:
            return None

        for part in link_header.split(","):
            part = part.strip()
            if 'rel="next"' not in part:
                continue

            start = part.find("<")
            end = part.find(">", start + 1)
            if start != -1 and end != -1:
                return part[start + 1:end]

        return None

    def _build_root_params(
        self,
        *,
        page: int,
        per_page: int,
        membership: bool,
        archived: bool,
        simple: bool,
    ) -> dict[str, Any]:
        """
        Build query parameters for offset-based root project listing requests.

        Args:
            page: One-based GitLab page number.
            per_page: Number of projects requested per page.
            membership: If True, include only projects the token user belongs to.
            archived: If True, include archived projects.
            simple: If True, request GitLab's simplified project payload.

        Returns:
            A dict of query parameters suitable for ``GET /projects``.
        """
        params: dict[str, Any] = {
            "page": int(page),
            "per_page": int(per_page),
            "order_by": "id",
            "sort": "desc",
            "membership": str(bool(membership)).lower(),
            "archived": str(bool(archived)).lower(),
        }
        if simple:
            params["simple"] = "true"
        return params

    def _build_tree_params(
        self,
        *,
        ref: str,
        page: int,
        per_page: int,
        subdir: str,
    ) -> dict[str, Any]:
        """
        Build query parameters for offset-based repository tree requests.

        Args:
            ref: Branch, tag, or commit SHA to list.
            page: One-based GitLab page number.
            per_page: Number of tree entries requested per page.
            subdir: Repository-internal directory path, or ``""`` for root.

        Returns:
            A dict of query parameters suitable for ``GET /repository/tree``.
        """
        params: dict[str, Any] = {
            "ref": ref,
            "page": int(page),
            "per_page": int(per_page),
        }
        if subdir:
            params["path"] = subdir
        return params

    def _normalize_projects(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Convert GitLab project payloads into filesystem project records.

        Args:
            data: Raw project dictionaries returned by GitLab.

        Returns:
            A list of dicts with ``id`` and ``original_path`` keys.
        """
        return [
            {
                "id": p["id"],
                "original_path": p["path_with_namespace"],
            }
            for p in data
        ]

    def _normalized_headers(self, headers) -> dict[str, str]:
        """
        Lowercase response header names for case-insensitive lookup.

        Args:
            headers: Mapping-like response headers from aiohttp.

        Returns:
            A plain ``dict[str, str]`` keyed by lowercase header names.
        """
        return {k.lower(): v for k, v in headers.items()}

    def _parse_total_count(self, headers) -> int | None:
        """
        Parse GitLab's ``X-Total`` pagination header.

        GitLab omits this header once a query would return more than 10,000
        records, so its absence is normal on large instances rather than an
        error.

        Args:
            headers: Mapping-like response headers from aiohttp.

        Returns:
            Total item count as ``int`` when GitLab supplies a numeric value,
            otherwise ``None``.
        """
        total_raw = (self._normalized_headers(headers).get("x-total") or "").strip()
        if not total_raw:
            return None
        try:
            return int(total_raw)
        except ValueError:
            return None

    def _has_next_page(self, headers, item_count: int, per_page: int) -> bool:
        """
        Report whether GitLab says another page follows this one.

        Both ``X-Next-Page`` and the ``Link`` header are consulted, because an
        instance may send either. When a server sends neither, the page is
        assumed to be full-means-more: under-reporting would tell a caller the
        listing ends here and hide every later entry, while over-reporting only
        costs one empty request.

        Args:
            headers: Mapping-like response headers from aiohttp.
            item_count: Number of entries returned in this page.
            per_page: Page size actually applied by the server.

        Returns:
            True when another page follows, otherwise False.
        """
        h = self._normalized_headers(headers)
        if (h.get("x-next-page") or "").strip():
            return True
        if "x-next-page" in h or "link" in h:
            return self._extract_next_link(h.get("link")) is not None

        warnings.warn(
            "list_page: no X-Total, X-Next-Page or Link header; assuming a full page "
            "means more entries follow",
            RuntimeWarning,
            stacklevel=2,
        )
        return item_count >= per_page

    def _applied_per_page(self, headers, per_page: int) -> int:
        """
        Return the page size GitLab actually applied.

        GitLab caps ``per_page`` and reports what it used in ``X-Per-Page``, so
        a request for more than the maximum silently receives a shorter page.
        Trusting the requested value would overstate how much has been seen.

        Args:
            headers: Mapping-like response headers from aiohttp.
            per_page: Page size that was requested.

        Returns:
            The applied page size, falling back to ``per_page``.
        """
        raw = (self._normalized_headers(headers).get("x-per-page") or "").strip()
        try:
            applied = int(raw)
        except ValueError:
            return per_page
        return applied if applied > 0 else per_page

    def _total_or_bound(self, headers, page: int, per_page: int, item_count: int) -> int:
        """
        Return GitLab's exact total, or a lower bound when it supplies none.

        An exact ``X-Total`` always wins. Otherwise the count of entries up to
        and including this page is used, plus one while another page follows.
        That trailing ``+1`` is load-bearing for callers that compare the total
        against what they received to decide whether more entries exist, so
        simplifying it away would make a truncated listing look complete.

        Args:
            headers: Mapping-like response headers from aiohttp.
            page: One-based page number just fetched.
            per_page: Page size that was requested.
            item_count: Number of entries returned in this page.

        Returns:
            The exact total when known, otherwise a lower bound.
        """
        exact = self._parse_total_count(headers)
        if exact is not None:
            return exact

        if not item_count:
            # An empty page means the requested offset is past the end. It says
            # nothing about how many entries exist, and ``(page - 1) * per_page``
            # would be the caller's own offset: an upper bound, not a lower one,
            # and unbounded in the offset the caller chose.
            return 0

        applied = self._applied_per_page(headers, per_page)
        seen = (page - 1) * applied + item_count
        return seen + 1 if self._has_next_page(headers, item_count, applied) else seen

    def _warn_offset_fallback(self, scope: str, reason: str) -> None:
        """
        Emit a runtime warning when offset listing must use its fallback path.

        Args:
            scope: Name of the caller reporting the fallback.
            reason: Human-readable reason the fallback path was selected.

        Returns:
            None.
        """
        warnings.warn(
            f"{scope}: using concurrent offset fallback ({reason})",
            RuntimeWarning,
            stacklevel=2,
        )

    async def _fetch_projects_page(
        self,
        *,
        page_num: int,
        per_page: int,
        membership: bool,
        archived: bool,
        simple: bool,
        allow_missing_page: bool = False,
    ) -> tuple[int, list[dict[str, Any]], dict[str, str]]:
        """
        Fetch one offset page of root projects.

        Args:
            page_num: One-based page number to request.
            per_page: Number of projects requested per page.
            membership: If True, include only projects the token user belongs to.
            archived: If True, include archived projects.
            simple: If True, request GitLab's simplified project payload.
            allow_missing_page: If True, treat GitLab 400/404 responses as an
                empty page for fallback probing.

        Returns:
            ``(page_num, items, headers)`` where ``items`` is a list of
            normalized project dicts and ``headers`` has lowercase keys.
        """
        s = await self._ensure()
        url = self._projects_url()
        params = self._build_root_params(
            page=page_num,
            per_page=per_page,
            membership=membership,
            archived=archived,
            simple=simple,
        )

        async with s.get(url, params=params) as r:
            if allow_missing_page and r.status in {400, 404}:
                return page_num, [], {}

            r.raise_for_status()
            data = await r.json()
            headers = self._normalized_headers(r.headers)
            return page_num, self._normalize_projects(data), headers

    async def _fetch_project_tree_page(
        self,
        repo_id: int,
        subdir: str,
        *,
        ref: str,
        page_num: int,
        per_page: int,
        allow_missing_page: bool = False,
    ) -> tuple[int, list[dict[str, Any]], dict[str, str]]:
        """
        Fetch one offset page of repository tree entries.

        Args:
            repo_id: Numeric GitLab project id.
            subdir: Repository-internal directory path, or ``""`` for root.
            ref: Branch, tag, or commit SHA to list.
            page_num: One-based page number to request.
            per_page: Number of tree entries requested per page.
            allow_missing_page: If True, treat GitLab 400/404 responses as an
                empty page for fallback probing.

        Returns:
            ``(page_num, items, headers)`` where ``items`` is a list of raw
            tree entry dicts and ``headers`` has lowercase keys.
        """
        s = await self._ensure()
        url = self._project_tree_url(repo_id)
        params = self._build_tree_params(
            ref=ref,
            page=page_num,
            per_page=per_page,
            subdir=subdir,
        )

        async with s.get(url, params=params) as r:
            if allow_missing_page and r.status in {400, 404}:
                return page_num, [], {}

            if r.status == 404:
                raise FileNotFoundError(subdir or str(repo_id))

            r.raise_for_status()
            data = await r.json()
            headers = self._normalized_headers(r.headers)
            return page_num, data, headers

    async def _collect_offset_pages(
        self,
        *,
        fetch_page: Callable[..., Awaitable[tuple[int, list[dict[str, Any]], dict[str, str]]]],
        first_page: tuple[int, list[dict[str, Any]], dict[str, str]],
        max_concurrency: int,
        scope: str,
    ) -> list[dict[str, Any]]:
        """
        Collect all items from an offset-paginated GitLab endpoint.

        Args:
            fetch_page: Callback that fetches one page by number.
            first_page: The already-fetched first page result.
            max_concurrency: Maximum number of pages fetched per chunk.
            scope: Caller name used for fallback warnings.

        Returns:
            The full combined item list in page order.
        """
        _, first_items, first_headers = first_page
        total_pages_raw = first_headers.get("x-total-pages") or ""
        chunk_size = max(1, min(int(max_concurrency), 4))

        if total_pages_raw:
            total_pages = int(total_pages_raw)
            if total_pages <= 1:
                return first_items

            out = list(first_items)
            next_page = 2

            while next_page <= total_pages:
                chunk_pages = range(
                    next_page,
                    min(total_pages + 1, next_page + chunk_size),
                )
                results = list(
                    await asyncio.gather(*(fetch_page(page_num) for page_num in chunk_pages))
                )
                results.sort(key=lambda x: x[0])

                for _, items, _ in results:
                    out.extend(items)

                next_page += chunk_size

            return out

        self._warn_offset_fallback(scope, "X-Total-Pages header unavailable")

        out = list(first_items)
        next_page = 2

        while True:
            chunk_pages = range(next_page, next_page + chunk_size)
            results = list(
                await asyncio.gather(
                    *(fetch_page(page_num, allow_missing_page=True) for page_num in chunk_pages)
                )
            )
            results.sort(key=lambda x: x[0])

            stop = False
            for _, items, _ in results:
                if not items:
                    stop = True
                    break
                out.extend(items)

            if stop:
                break

            next_page += chunk_size

        return out

    async def _collect_keyset_pages(
        self,
        first_url: str,
        *,
        params: dict[str, Any] | None = None,
        item_normalizer: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
        not_found_error_factory: Callable[[], Exception] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Collect all items from a sequential keyset-paginated GitLab endpoint.

        Args:
            first_url: URL for the initial request.
            params: Optional query params for the initial request only.
            item_normalizer: Callback applied to each page of JSON items.
            not_found_error_factory: Optional factory for custom 404 handling.

        Returns:
            The full combined item list in traversal order.
        """
        s = await self._ensure()
        out: list[dict[str, Any]] = []
        next_url: str | None = first_url
        first_request = True

        while next_url:
            request_kwargs: dict[str, Any] = {}
            if first_request and params is not None:
                request_kwargs["params"] = params

            async with s.get(next_url, **request_kwargs) as r:
                if r.status == 404 and not_found_error_factory is not None:
                    raise not_found_error_factory()

                r.raise_for_status()
                data = await r.json()
                out.extend(item_normalizer(data))
                next_url = self._extract_next_link(r.headers.get("Link"))

            first_request = False

        return out

    # ------------------------------------------------------------------
    # Project lookup / metadata
    # ------------------------------------------------------------------
    async def get_project_by_path(
        self,
        path_with_namespace: str,
    ) -> dict[str, Any] | None:
        """
        Look up a project by its full GitLab ``path_with_namespace``.

        Args:
            path_with_namespace: Project path such as ``"group/sub/repo"``.

        Returns:
            A minimal project dict with ``id`` and ``original_path`` when the
            project exists, otherwise ``None`` for GitLab 404 responses.
        """
        s = await self._ensure()
        url = self._project_by_path_url(path_with_namespace)

        async with s.get(url) as r:
            if r.status == 404:
                return None
            r.raise_for_status()
            j = await r.json()
            return {"id": j["id"], "original_path": j["path_with_namespace"]}

    async def get_project_by_id(self, repo_id: int) -> dict[str, Any]:
        """
        Retrieve project metadata by numeric GitLab project id.

        Args:
            repo_id: Numeric GitLab project id.

        Returns:
            A project dict containing ``id``, ``original_path``, and
            ``default_branch``.
        """
        s = await self._ensure()
        url = self._project_url(repo_id)

        async with s.get(url) as r:
            if r.status == 404:
                raise FileNotFoundError(f"Project id {repo_id} not found")
            r.raise_for_status()
            j = await r.json()
            return {
                "id": j["id"],
                "original_path": j["path_with_namespace"],
                "default_branch": j.get("default_branch") or "main",
            }

    async def get_default_branch(self, repo_id: int) -> str:
        """
        Return the project's default branch name.

        Args:
            repo_id: Numeric GitLab project id.

        Returns:
            Default branch name as ``str``.

        Falls back to ``"main"`` if GitLab does not report a default branch.
        Raises ``FileNotFoundError`` if the project does not exist.
        """
        project = await self.get_project_by_id(repo_id)
        return project.get("default_branch") or "main"

    # ------------------------------------------------------------------
    # Root listing
    # ------------------------------------------------------------------
    async def _retrieve_root_level_keyset_sequential(
            self,
            *,
            per_page: int = 100,
            membership: bool = False,
            archived: bool = False,
            simple: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Sequential keyset-based full project listing via GET /projects.

        This is a full-list fallback builder, not a random-access page API.

        Args:
            per_page: Number of projects requested per keyset page.
            membership: If True, include only projects the token user belongs to.
            archived: If True, include archived projects.
            simple: If True, request GitLab's simplified project payload.

        Returns:
            A list of normalized project dicts with ``id`` and
            ``original_path`` keys.
        """
        if per_page < 1:
            raise ValueError("per_page must be >= 1")

        params: dict[str, Any] = {
            "pagination": "keyset",
            "per_page": int(per_page),
            "order_by": "id",
            "sort": "asc",
            "membership": str(bool(membership)).lower(),
            "archived": str(bool(archived)).lower(),
        }
        if simple:
            params["simple"] = "true"

        return await self._collect_keyset_pages(
            self._projects_url(),
            params=params,
            item_normalizer=self._normalize_projects,
        )

    async def retrieve_root_level(
            self,
            *,
            per_page: int = 100,
            membership: bool = False,
            archived: bool = False,
            simple: bool = True,
            max_concurrency: int = 4,
    ) -> list[dict[str, Any]]:
        """
        Full project listing via GET /projects.

        Strategy:
          - try offset-based full fetch first
          - use concurrent chunk fetching where possible
          - fall back to sequential keyset-based full fetch on failure

        Args:
            per_page: Number of projects requested per page.
            membership: If True, include only projects the token user belongs to.
            archived: If True, include archived projects.
            simple: If True, request GitLab's simplified project payload.
            max_concurrency: Maximum number of offset pages fetched per chunk;
                internally capped to avoid overly broad concurrent requests.

        Returns:
            A list of normalized project dicts with ``id`` and
            ``original_path`` keys.
        """
        if per_page < 1:
            raise ValueError("per_page must be >= 1")

        try:
            first_page = await self._fetch_projects_page(
                page_num=1,
                per_page=per_page,
                membership=membership,
                archived=archived,
                simple=simple,
            )
            return await self._collect_offset_pages(
                fetch_page=lambda page_num, *, allow_missing_page=False: self._fetch_projects_page(
                    page_num=page_num,
                    per_page=per_page,
                    membership=membership,
                    archived=archived,
                    simple=simple,
                    allow_missing_page=allow_missing_page,
                ),
                first_page=first_page,
                max_concurrency=max_concurrency,
                scope="retrieve_root_level",
            )

        except Exception:
            warnings.warn(
                "retrieve_root_level: falling back to sequential keyset listing",
                RuntimeWarning,
                stacklevel=2,
            )
            return await self._retrieve_root_level_keyset_sequential(
                per_page=per_page,
                membership=membership,
                archived=archived,
                simple=simple,
            )

    async def retrieve_root_level_page(
        self,
        *,
        page: int = 1,
        per_page: int = 100,
        membership: bool = False,
        archived: bool = False,
        simple: bool = True,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Retrieve exactly one offset-based page from /projects.

        Args:
            page: One-based GitLab page number.
            per_page: Number of projects requested for the page.
            membership: If True, include only projects the token user belongs to.
            archived: If True, include archived projects.
            simple: If True, request GitLab's simplified project payload.

        Returns:
            ``(items, total_count)`` where ``items`` is a list of normalized
            project dicts and ``total_count`` is GitLab's ``X-Total`` when it
            supplies one, otherwise a lower bound covering the pages seen so
            far. See ``_total_or_bound``.
        """
        if page < 1:
            raise ValueError("page must be >= 1")
        if per_page < 1:
            raise ValueError("per_page must be >= 1")

        s = await self._ensure()
        url = self._projects_url()
        params = self._build_root_params(
            page=page,
            per_page=per_page,
            membership=membership,
            archived=archived,
            simple=simple,
        )

        async with s.get(url, params=params) as r:
            r.raise_for_status()
            data = await r.json()
            items = self._normalize_projects(data)
            total_count = self._total_or_bound(r.headers, page, per_page, len(items))
            return items, total_count

    # ------------------------------------------------------------------
    # Project directory listing
    # ------------------------------------------------------------------

    async def _retrieve_project_level_keyset_sequential(
            self,
            repo_id: int,
            subdir: str,
            *,
            ref: str | None = None,
            per_page: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Sequential keyset-based full directory listing via
        GET /projects/:id/repository/tree.

        This is a full-list fallback builder, not a random-access page API.

        Args:
            repo_id: Numeric GitLab project id.
            subdir: Repository-internal directory path, or ``""`` for root.
            ref: Branch, tag, or commit SHA to list. If ``None``, the project
                default branch is used.
            per_page: Number of tree entries requested per keyset page.

        Returns:
            A list of raw GitLab repository tree entry dicts.
        """
        if per_page < 1:
            raise ValueError("per_page must be >= 1")

        if ref is None:
            ref = await self.get_default_branch(repo_id)

        params: dict[str, Any] = {
            "ref": ref,
            "per_page": int(per_page),
            "pagination": "keyset",
        }
        if subdir:
            params["path"] = subdir

        return await self._collect_keyset_pages(
            self._project_tree_url(repo_id),
            params=params,
            item_normalizer=lambda data: data,
            not_found_error_factory=lambda: FileNotFoundError(subdir or str(repo_id)),
        )

    async def retrieve_project_level(
            self,
            repo_id: int,
            subdir: str,
            *,
            ref: str | None = None,
            per_page: int = 100,
            max_concurrency: int = 4,
    ) -> list[dict[str, Any]]:
        """
        Full directory listing via GET /projects/:id/repository/tree.

        Strategy:
          - try offset-based full fetch first
          - use concurrent chunk fetching where possible
          - fall back to sequential keyset-based full fetch on failure

        Args:
            repo_id: Numeric GitLab project id.
            subdir: Repository-internal directory path, or ``""`` for root.
            ref: Branch, tag, or commit SHA to list. If ``None``, the project
                default branch is used.
            per_page: Number of tree entries requested per page.
            max_concurrency: Maximum number of offset pages fetched per chunk;
                internally capped to avoid overly broad concurrent requests.

        Returns:
            A list of raw GitLab repository tree entry dicts.
        """
        if per_page < 1:
            raise ValueError("per_page must be >= 1")

        if ref is None:
            ref = await self.get_default_branch(repo_id)

        try:
            first_page = await self._fetch_project_tree_page(
                repo_id,
                subdir,
                ref=ref,
                page_num=1,
                per_page=per_page,
            )
            return await self._collect_offset_pages(
                fetch_page=lambda page_num, *, allow_missing_page=False: self._fetch_project_tree_page(
                    repo_id,
                    subdir,
                    ref=ref,
                    page_num=page_num,
                    per_page=per_page,
                    allow_missing_page=allow_missing_page,
                ),
                first_page=first_page,
                max_concurrency=max_concurrency,
                scope="retrieve_project_level",
            )

        except FileNotFoundError:
            raise
        except Exception:
            warnings.warn(
                "retrieve_project_level: falling back to sequential keyset listing",
                RuntimeWarning,
                stacklevel=2,
            )
            return await self._retrieve_project_level_keyset_sequential(
                repo_id,
                subdir,
                ref=ref,
                per_page=per_page,
            )

    async def retrieve_project_level_page(
        self,
        repo_id: int,
        subdir: str,
        *,
        ref: str | None = None,
        page: int = 1,
        per_page: int = 100,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Retrieve exactly one offset-based page from /repository/tree.

        Args:
            repo_id: Numeric GitLab project id.
            subdir: Repository-internal directory path, or ``""`` for root.
            ref: Branch, tag, or commit SHA to list. If ``None``, the project
                default branch is used.
            page: One-based GitLab page number.
            per_page: Number of tree entries requested for the page.

        Returns:
            ``(items, total_count)`` where ``items`` is a list of raw tree
            entry dicts and ``total_count`` is GitLab's ``X-Total`` when it
            supplies one, otherwise a lower bound covering the pages seen so
            far. See ``_total_or_bound``.
        """
        if page < 1:
            raise ValueError("page must be >= 1")
        if per_page < 1:
            raise ValueError("per_page must be >= 1")

        if ref is None:
            ref = await self.get_default_branch(repo_id)

        s = await self._ensure()
        url = self._project_tree_url(repo_id)
        params = self._build_tree_params(
            ref=ref,
            page=page,
            per_page=per_page,
            subdir=subdir,
        )

        async with s.get(url, params=params) as r:
            if r.status == 404:
                raise FileNotFoundError(subdir or str(repo_id))
            r.raise_for_status()

            items = await r.json()
            total_count = self._total_or_bound(r.headers, page, per_page, len(items))
            return items, total_count

    # ------------------------------------------------------------------
    # Read / file metadata
    # ------------------------------------------------------------------
    async def stream_file(
        self,
        repo_id: int,
        path: str,
        ref: str,
        *,
        chunk_size: int = 1024 * 1024,
    ):
        """
        Stream raw file content from GitLab in chunks.

        Passes ``lfs=true`` so GitLab resolves LFS objects.

        Args:
            repo_id: Numeric GitLab project id.
            path: Repository-internal file path.
            ref: Branch, tag, or commit SHA to read from.
            chunk_size: Maximum bytes requested from aiohttp per yielded chunk.

        Yields:
            ``bytes`` chunks of file content.
        """
        s = await self._ensure()
        url = self._repository_file_raw_url(repo_id, path)

        async with s.get(url, params={"ref": ref, "lfs": "true"}) as r:
            if r.status == 404:
                raise FileNotFoundError(path)
            r.raise_for_status()

            async for chunk in r.content.iter_chunked(chunk_size):
                if chunk:
                    yield chunk

    async def get_file(self, repo_id: int, path: str, ref: str) -> dict[str, Any]:
        """
        Retrieve repository file metadata/content via Repository Files API.

        Args:
            repo_id: Numeric GitLab project id.
            path: Repository-internal file path.
            ref: Branch, tag, or commit SHA to read from.

        Returns:
            Raw GitLab repository file JSON as a dict.

        Raises:
            RefNotFound: If GitLab could not resolve ``ref``. A subclass of
                ``FileNotFoundError``, so catching that alone still works.
            FileNotFoundError: If ``ref`` resolved but does not contain ``path``.

        Content is base64-encoded in GitLab's response.
        """
        s = await self._ensure()
        url = self._repository_file_url(repo_id, path)

        async with s.get(url, params={"ref": ref}) as r:
            if r.status == 404:
                # GitLab answers 404 both for a path that is not in the tree and for a ref it
                # cannot resolve, and says which in the body. Callers choose between creating
                # and updating a file on this answer, so reporting a missing ref as a missing
                # file sends them on to commit against a branch that is not there.
                if "Commit Not Found" in await _gitlab_message(r):
                    raise RefNotFound(f"Ref {ref!r} could not be resolved in project {repo_id}")
                raise FileNotFoundError(path)
            r.raise_for_status()
            return await r.json()

    # ------------------------------------------------------------------
    # Git write / mutation
    # ------------------------------------------------------------------
    async def create_branch(self, repo_id: int, branch: str, ref: str) -> None:
        """
        Create a branch from an existing ref.

        Args:
            repo_id: Numeric GitLab project id.
            branch: Name of the branch to create.
            ref: Source branch, tag, or commit SHA.

        Returns:
            None.

        If the branch already exists, treat that as success.
        """
        s = await self._ensure()
        url = self._branches_url(repo_id)

        async with s.post(url, data={"branch": branch, "ref": ref}) as r:
            if r.status < 400:
                return
            # This is the first write of the transaction, so a refusal here is what a caller
            # actually meets: a protected branch, a missing source ref, a token that cannot push.
            # raise_for_status keeps only "Bad Request", so without reading the body the reason
            # is gone one frame before create_commit, which already does read it.
            text = await r.text()
            # Read the raw text for this, not the parsed message: an instance behind a proxy can
            # answer with something that is not JSON, and treating "already exists" as a failure
            # would turn a second upload with the same token into an error.
            if r.status == 400 and "already exists" in text.lower():
                return
            message = _message_from_text(text)
            if message:
                raise aiohttp.ClientResponseError(
                    r.request_info,
                    r.history,
                    status=r.status,
                    message=message,
                    headers=r.headers,
                )
            r.raise_for_status()

    async def create_commit(
        self,
        repo_id: int,
        branch: str,
        commit_message: str,
        actions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Create a commit with one or more file actions.

        Args:
            repo_id: Numeric GitLab project id.
            branch: Branch that receives the commit.
            commit_message: Commit message sent to GitLab.
            actions: GitLab commit action dicts such as create, update, or move.

        Returns:
            Raw GitLab commit response JSON as a dict.

        Raises:
            aiohttp.ClientResponseError: If GitLab refuses the commit. Its ``message``
                carries GitLab's own reason when the response supplies one.
        """
        s = await self._ensure()
        url = self._commits_url(repo_id)

        payload = {
            "branch": branch,
            "commit_message": commit_message,
            "actions": actions,
        }

        async with s.post(url, json=payload) as r:
            # GitLab funnels every reason a commit was refused into one status and puts the
            # difference in the body: a protected branch, a stale last_commit_id and a path
            # that already exists all arrive as 400 Bad Request. raise_for_status keeps only
            # the status and the reason phrase, so the one thing telling them apart is lost
            # before the caller ever sees it.
            message = await _gitlab_message(r) if r.status >= 400 else ""
            if message:
                raise aiohttp.ClientResponseError(
                    r.request_info,
                    r.history,
                    status=r.status,
                    message=message,
                    headers=r.headers,
                )
            r.raise_for_status()
            return await r.json()

    async def _find_open_merge_request(
        self,
        repo_id: int,
        source_branch: str,
        target_branch: str,
    ) -> dict[str, Any] | None:
        """Return the open merge request matching both branches, if any."""
        s = await self._ensure()
        url = self._merge_requests_url(repo_id)
        params = {
            "state": "opened",
            "source_branch": source_branch,
            "target_branch": target_branch,
        }
        async with s.get(url, params=params) as r:
            r.raise_for_status()
            merge_requests = await r.json()

        for merge_request in merge_requests:
            if (
                merge_request.get("state") == "opened"
                and merge_request.get("source_branch") == source_branch
                and merge_request.get("target_branch") == target_branch
            ):
                return merge_request
        return None

    async def ensure_merge_request(
        self,
        repo_id: int,
        source_branch: str,
        target_branch: str,
        title: str,
    ) -> dict[str, Any]:
        """
        Ensure a merge request exists for a source branch and target branch.

        Args:
            repo_id: Numeric GitLab project id.
            source_branch: Branch containing the proposed changes.
            target_branch: Branch that should receive the changes.
            title: Merge request title.

        Returns:
            The matching open merge request, whether it already existed or was
            created by this call.
        """
        existing = await self._find_open_merge_request(
            repo_id,
            source_branch,
            target_branch,
        )
        if existing is not None:
            return existing

        s = await self._ensure()
        url = self._merge_requests_url(repo_id)

        payload = {
            "source_branch": source_branch,
            "target_branch": target_branch,
            "title": title,
        }

        conflict_error: aiohttp.ClientResponseError | None = None
        async with s.post(url, json=payload) as r:
            if r.status != 409:
                r.raise_for_status()
                return await r.json()
            try:
                r.raise_for_status()
            except aiohttp.ClientResponseError as error:
                conflict_error = error

        existing = await self._find_open_merge_request(
            repo_id,
            source_branch,
            target_branch,
        )
        if existing is not None:
            return existing

        if conflict_error is not None:
            raise conflict_error
        raise RuntimeError("GitLab returned HTTP 409 without an error response")

    async def create_merge_request(
        self,
        repo_id: int,
        source_branch: str,
        target_branch: str,
        title: str,
    ) -> dict[str, Any]:
        """
        Compatibility wrapper for :meth:`ensure_merge_request`.

        Retained for compatibility; new code should use
        ``ensure_merge_request()``.
        """
        return await self.ensure_merge_request(
            repo_id=repo_id,
            source_branch=source_branch,
            target_branch=target_branch,
            title=title,
        )

    # ------------------------------------------------------------------
    # Git LFS
    # ------------------------------------------------------------------
    async def lfs_batch(
            self,
            namespace: str,
            token: str,
            payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Call the Git LFS batch endpoint for upload negotiation.

        Args:
            namespace: Project ``path_with_namespace`` used in the LFS URL.
            token: Token used for Git LFS basic authentication.
            payload: Git LFS batch request payload.

        Returns:
            Raw Git LFS batch response JSON as a dict.
        """
        s = await self._ensure()
        url = self._lfs_batch_url(namespace)

        headers = {
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
        }

        auth = aiohttp.BasicAuth("oauth2", token) if token else None

        async with s.post(url, json=payload, headers=headers, auth=auth) as r:
            r.raise_for_status()
            return await r.json()

    async def lfs_upload(
        self,
        token: str,
        href: str,
        headers: dict[str, str],
        data_stream,
        *,
        chunk_size: int = 1024 * 1024,
    ) -> None:
        """
        Upload the binary LFS object to the negotiated upload URL.

        Args:
            token: Token used for Git LFS basic authentication when the
                negotiated headers do not already include authorization.
            href: Negotiated upload URL returned by the LFS batch endpoint.
            headers: Headers returned by the LFS batch endpoint for upload.
            data_stream: Async readable binary stream, already seeked to the
                beginning by the caller.
            chunk_size: Maximum bytes read from ``data_stream`` per chunk.

        Returns:
            None.
        """
        s = await self._ensure()

        async def gen():
            """
            Yield chunks from the async data stream for aiohttp upload.

            Yields:
                ``bytes`` chunks read from ``data_stream``.
            """
            while True:
                chunk = await data_stream.read(chunk_size)
                if not chunk:
                    break
                yield chunk

        req_headers = dict(headers or {})
        for name in list(req_headers):
            if name.lower() == "transfer-encoding":
                req_headers.pop(name)

        has_authorization_header = any(
            k.lower() == "authorization" for k in req_headers
        )

        auth = None
        if token and not has_authorization_header:
            auth = aiohttp.BasicAuth("oauth2", token)

        async with s.put(href, data=gen(), headers=req_headers, auth=auth) as r:
            r.raise_for_status()

    async def upload_file_lfs(
        self,
        *,
        token: str,
        repo: dict[str, Any],
        final_path: str,
        local_path: str,
        base_branch: Optional[str] = None,
        feature_branch: str,
        create_mr: bool = True,
        mode: str = "overwrite",
    ) -> str:
        """
        Upload a local file as Git LFS by default.

        Args:
            token: Token used for Git LFS authentication.
            repo: Project dict containing at least ``id`` and ``original_path``.
            final_path: Repository-internal destination path for the LFS pointer.
            local_path: Local filesystem path to the file being uploaded.
            base_branch: Branch to base the feature branch on. If ``None``, the
                project default branch is used.
            feature_branch: Complete feature branch name.
            create_mr: If True, create a merge request after committing.
            mode: ``"create"`` refuses an existing target; ``"overwrite"``
                uses the ARCfs safe replacement policy.

        Returns:
            Created feature branch name as ``str``.
        """
        sha = await calculate_sha256(local_path)

        import aiofiles

        async with aiofiles.open(local_path, "rb") as f:
            await f.seek(0, 2)
            size = await f.tell()
            await f.seek(0)

            return await commit_lfs_transaction(
                client=self,
                token=token,
                repo=repo,
                base_branch=base_branch,
                final_path=final_path,
                sha=sha,
                size=size,
                data_stream=f,
                feature_branch=feature_branch,
                tmp_pointer_name=True,
                create_mr=create_mr,
                mode=mode,
            )

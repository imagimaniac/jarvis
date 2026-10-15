/* Jarvis service worker: application shell only.
 *
 * The single rule this file exists to enforce is that personal data never
 * reaches the Cache API. Everything under /api/ is memory, search results,
 * captured entries and auth failures, so those requests are handed straight back
 * to the network without ever calling respondWith(). Not caching them is the
 * whole point; a cache hit would survive a logout and keep serving one user's
 * data to the next person who opens the app.
 *
 * Offline support is therefore limited to "the page loads and looks right".
 * Jarvis cannot answer a query with the radio off, and it should fail visibly
 * rather than show a stale or empty-looking result.
 */

const VERSION = 'v2';
const SHELL_CACHE = 'jarvis-shell-' + VERSION;

/* Only what is needed to render the app with no network. Notably absent:
 * /api/*, /sw.js, and anything reachable with a token in the URL. */
const SHELL_ASSETS = [
  '/',
  '/manifest.json',
  '/icon-192.png',
  '/icon-512.png',
  '/maskable-512.png',
  '/apple-touch-icon.png'
];

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(SHELL_CACHE);
    await cache.addAll(SHELL_ASSETS);
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    // Drop every older shell so a new VERSION actually replaces the old one.
    const keys = await caches.keys();
    await Promise.all(
      keys.filter((k) => k.startsWith('jarvis-shell-') && k !== SHELL_CACHE)
          .map((k) => caches.delete(k))
    );
    await self.clients.claim();
  })());
});

/* Never cache a request the user authenticated or that carries a credential. */
function isPrivate(req, url) {
  if (url.pathname === '/api' || url.pathname.startsWith('/api/')) return true;
  if (url.searchParams.has('token')) return true;
  if (req.headers.has('authorization')) return true;
  if (req.headers.has('cookie')) return true;
  return false;
}

self.addEventListener('fetch', (event) => {
  const req = event.request;

  // Writes (add/delete) are never intercepted.
  if (req.method !== 'GET') return;

  let url;
  try {
    url = new URL(req.url);
  } catch (e) {
    return;
  }

  // Same-origin only; anything else is left entirely alone.
  if (url.origin !== self.location.origin) return;

  // Personal data and credentials bypass the cache completely.
  if (isPrivate(req, url)) return;

  // Navigations: network first so the app is never stale, cached shell as the
  // offline fallback. This is also what keeps a bootstrap response, which is
  // served no-store, from being written into the shell cache.
  if (req.mode === 'navigate') {
    event.respondWith((async () => {
      try {
        const fresh = await fetch(req);
        if (fresh && fresh.ok && fresh.type === 'basic') {
          const cache = await caches.open(SHELL_CACHE);
          cache.put('/', fresh.clone());
        }
        return fresh;
      } catch (err) {
        const cached = await caches.match('/');
        return cached || Response.error();
      }
    })());
    return;
  }

  // Remaining shell assets: cache first, then fill in on first use.
  event.respondWith((async () => {
    const cached = await caches.match(req);
    if (cached) return cached;
    const res = await fetch(req);
    // type 'basic' excludes opaque and cross-origin responses, which cannot be
    // safely cached; only successes are stored.
    if (res && res.ok && res.type === 'basic') {
      const cache = await caches.open(SHELL_CACHE);
      cache.put(req, res.clone());
    }
    return res;
  })());
});

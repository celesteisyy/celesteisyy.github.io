"""Exercise the installed static routes with a real Nginx process."""
import http.client
import shutil
import socket
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


class StaticRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="cyssie-nginx-")
        cls.directory = Path(cls.temp.name)
        cls.directory.chmod(0o755)
        static = cls.directory / "static"
        static.mkdir()
        for name in ("index.html", "embed.html", "sign-in.html", "login-complete.html", "embed.js", "webui.css"):
            (static / name).write_text("fixture " + name)
        (static / "private.env").write_text("must not be served")
        snippet = (Path(__file__).parents[1] / "deploy/nginx.conf").read_text()
        snippet = snippet.replace("/var/www/cyth-playground", str(static))
        guard = (Path(__file__).parents[1] / "deploy/origin-guard.conf").read_text()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            cls.port = listener.getsockname()[1]
        config = cls.directory / "nginx.conf"
        config.write_text(
            "pid " + str(cls.directory / "nginx.pid") + ";\n"
            "error_log " + str(cls.directory / "error.log") + " notice;\n"
            "events {}\nhttp { access_log off;\n"
            + "".join(f"{kind}_temp_path {cls.directory / kind};\n" for kind in
                      ("client_body", "proxy", "fastcgi", "uwsgi", "scgi"))
            + guard + "\nserver {\n"
            f"listen 127.0.0.1:{cls.port};\n"
            + snippet + "\nlocation / { return 418; }\n} }\n"
        )
        cls.process = subprocess.Popen(
            [shutil.which("nginx") or "nginx", "-p", str(cls.directory), "-c", str(config),
             "-g", "daemon off; master_process off;"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        for _ in range(100):
            if cls.process.poll() is not None:
                raise RuntimeError(cls.process.stderr.read().decode())
            try:
                cls.get("/")
                return
            except OSError:
                time.sleep(0.03)
        raise RuntimeError("Nginx did not start")

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        cls.process.communicate(timeout=5)
        cls.temp.cleanup()

    @classmethod
    def get(cls, path, method="GET", headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", cls.port, timeout=3)
        try:
            connection.request(method, path, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_landing_and_index_serve_html_without_redirect_cycle(self):
        for path in ("/playground/", "/playground/index.html", "/playground/?from=site"):
            with self.subTest(path=path):
                status, headers, body = self.get(path)
                self.assertEqual(status, 200)
                self.assertEqual(body, b"fixture index.html")
                self.assertEqual(headers["Content-Type"], "text/html")
                self.assertEqual(headers["Cache-Control"], "no-store")
                self.assertNotIn("X-Frame-Options", headers)
                self.assertEqual(headers["Content-Security-Policy"], "frame-ancestors 'self' https://celesteisyy.github.io")
        status, _, body = self.get("/playground/", "HEAD")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"")

    def test_assets_are_exact_and_have_explicit_types(self):
        for name, mime in (("embed.js", "application/javascript"), ("webui.css", "text/css")):
            status, headers, body = self.get("/playground/" + name + "?v=test")
            self.assertEqual(status, 200)
            self.assertEqual(headers["Content-Type"], mime)
            self.assertEqual(body, ("fixture " + name).encode())

    def test_missing_slash_redirects_once(self):
        status, headers, _ = self.get("/playground")
        self.assertEqual(status, 302)
        self.assertTrue(headers["Location"].endswith("/playground/"))

    def test_other_files_and_existing_api_routes_are_not_captured(self):
        for path in ("/playground/private.env", "/playground/nested/index.html",
                     "/api/v1/auths/", "/api/models", "/oauth/oidc/callback", "/"):
            with self.subTest(path=path):
                status, _, body = self.get(path)
                self.assertEqual(status, 418)
                self.assertNotIn(b"must not be served", body)


    def test_embed_and_login_documents_are_explicit_static_routes(self):
        for route, name in (("/playground/embed/", "embed.html"),
                            ("/playground/embed.html", "embed.html"),
                            ("/playground/sign-in", "sign-in.html"),
                            ("/playground/login-complete", "login-complete.html")):
            status, headers, body = self.get(route)
            self.assertEqual(status, 200)
            self.assertEqual(body, ("fixture " + name).encode())
            self.assertEqual(headers["Content-Security-Policy"],
                             "frame-ancestors 'self' https://celesteisyy.github.io")

    def test_browser_writes_require_the_exact_webui_origin(self):
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            for origin in (None, "null", "https://evil.example",
                           "https://celesteisyy.github.io",
                           "https://cyth.18-118-248-231.sslip.io.evil.example"):
                headers = {"Origin": origin} if origin else {}
                headers["Cookie"] = "token=fixture"
                with self.subTest(method=method, origin=origin):
                    self.assertEqual(self.get("/api/tool-fixture", method, headers)[0], 403)
            self.assertEqual(self.get("/api/tool-fixture", method,
                {"Origin": "https://cyth.18-118-248-231.sslip.io"})[0], 418)

    def test_websocket_upgrades_require_the_exact_webui_origin(self):
        for origin in (None, "null", "https://evil.example", "https://celesteisyy.github.io"):
            headers = {"Upgrade": "websocket", "Connection": "Upgrade"}
            if origin: headers["Origin"] = origin
            self.assertEqual(self.get("/ws/socket.io/", headers=headers)[0], 403)
        self.assertEqual(self.get("/ws/socket.io/", headers={
            "Upgrade": "websocket", "Connection": "Upgrade",
            "Origin": "https://cyth.18-118-248-231.sslip.io"})[0], 418)


if __name__ == "__main__":
    unittest.main()

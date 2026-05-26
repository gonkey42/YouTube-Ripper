import unittest
from unittest.mock import patch

import app as webapp


class AccessControlTests(unittest.TestCase):
    def setUp(self):
        webapp.app.testing = True
        self.client = webapp.app.test_client()

    def get_from(self, remote_addr: str):
        return self.client.get("/", environ_base={"REMOTE_ADDR": remote_addr})

    def test_allows_ipv4_loopback(self):
        response = self.get_from("127.0.0.1")

        self.assertEqual(response.status_code, 200)

    def test_allows_ipv6_loopback(self):
        response = self.get_from("::1")

        self.assertEqual(response.status_code, 200)

    def test_allows_tailscale_ipv4(self):
        response = self.get_from("100.116.12.34")

        self.assertEqual(response.status_code, 200)

    def test_allows_tailscale_ipv6(self):
        response = self.get_from("fd7a:115c:a1e0::1234")

        self.assertEqual(response.status_code, 200)

    def test_rejects_regular_lan_client(self):
        response = self.get_from("192.168.1.25")

        self.assertEqual(response.status_code, 403)

    def test_allowed_clients_env_override_allows_custom_range(self):
        with patch.dict("os.environ", {"YOUTUBE_RIPPER_ALLOWED_CLIENTS": "192.168.1.0/24"}):
            response = self.get_from("192.168.1.25")

        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()

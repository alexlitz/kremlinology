"""Proxy-mode TLS witness: attest responses from any HTTPS API without seeing the caller's key.

The relay forwards raw TCP (it never terminates TLS) and records every TLS record with a
timestamp. After the session the client discloses only the server-to-client TLS 1.3 traffic
secrets; the relay verifies the server's certificate chain and CertificateVerify signature,
decrypts the response it recorded itself, and signs a receipt. Client-to-server traffic (the
request, including Authorization headers) stays encrypted under keys the relay never learns.
"""

#!/usr/bin/env python3
# Robustly dump SPKI public key bytes and certificate signature bytes
# Works even when the algorithm OID is unknown to libraries.
import sys, base64, re
from pyasn1.codec.der import decoder
from pyasn1.type import univ

def load_der(path):
    data = open(path, 'rb').read()
    if b'-----BEGIN' in data:
        # PEM -> DER
        b64 = b''.join(line.strip() for line in data.splitlines()
                       if not line.startswith(b'-----') and line.strip())
        return base64.b64decode(b64)
    return data

def maybe_unwrap_octet_string(payload: bytes) -> bytes:
    # If payload is a DER OCTET STRING (tag 0x04), unwrap it to its contents.
    if not payload:
        return payload
    if payload[0] != 0x04:
        return payload
    # Parse DER length
    if len(payload) < 2:
        return payload
    if payload[1] < 0x80:
        l = payload[1]
        hdr_len = 2
    else:
        nlen = payload[1] & 0x7F
        if len(payload) < 2 + nlen:
            return payload
        l = 0
        for i in range(nlen):
            l = (l << 8) | payload[2 + i]
        hdr_len = 2 + nlen
    if len(payload) < hdr_len + l:
        return payload
    return payload[hdr_len:hdr_len + l]

def main(cert_path):
    der = load_der(cert_path)
    cert, rest = decoder.decode(der)
    if rest:
        print(f"Warning: trailing bytes after certificate: {len(rest)}", file=sys.stderr)

    # Certificate ::= SEQUENCE { tbsCertificate, signatureAlgorithm, signatureValue }
    tbs = cert[0]
    sig_alg = cert[1]
    sig_val = cert[2]    # BIT STRING

    # TBSCertificate indices (v3): [0]=[0] EXPLICIT Version, [1]=serial, [2]=signature, [3]=issuer,
    # [4]=validity, [5]=subject, [6]=subjectPublicKeyInfo
    spki = tbs[6]
    spki_alg = spki[0]   # AlgorithmIdentifier ::= SEQUENCE { algorithm OBJECT IDENTIFIER, parameters ANY OPTIONAL }
    spki_pub = spki[1]   # BIT STRING

    spki_alg_oid = str(spki_alg[0])
    sig_alg_oid = str(sig_alg[0])

    # Extract bit string bytes (pyasn1 BitString.asOctets() returns content bits as octets)
    pk_bits = bytes(spki_pub.asOctets())
    sig_bytes = bytes(sig_val.asOctets())

    # Some PQC profiles wrap the key inside an OCTET STRING inside the BIT STRING; unwrap if so.
    pk_unwrapped = maybe_unwrap_octet_string(pk_bits)

    # Write outputs
    with open('pk.bin', 'wb') as f:
        f.write(pk_unwrapped)
    with open('sig.bin', 'wb') as f:
        f.write(sig_bytes)

    print(f"SubjectPublicKeyInfo.algorithm OID: {spki_alg_oid}")
    print(f"SignatureAlgorithm OID:            {sig_alg_oid}")
    print(f"Public key length (pk.bin):        {len(pk_unwrapped)} bytes (raw key payload)")
    print(f"Certificate signature length:      {len(sig_bytes)} bytes")
    print("Wrote pk.bin and sig.bin")

    # Quick size-based hinting for Dilithium (if it matches common sizes)
    sizes = {
        'Dilithium2': (1312, 2420),
        'Dilithium3': (1952, 3309),
        'Dilithium5': (2592, 4595),
    }
    hints = []
    for name, (pkL, sigL) in sizes.items():
        score = (len(pk_unwrapped) == pkL) + (len(sig_bytes) == sigL)
        if score:
            hints.append((score, name, pkL, sigL))
    if hints:
        hints.sort(reverse=True)
        print("Size hints:")
        for score, name, pkL, sigL in hints:
            print(f"  - {name}: pk={pkL}, sig={sigL} (matches: {score}/2)")
    else:
        print("No direct Dilithium size match; post the lengths and we’ll map to a scheme/parameter set.")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python dump_cert_generic.py quantumvillage.cert")
        sys.exit(1)
    main(sys.argv[1])
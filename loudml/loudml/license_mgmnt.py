import argparse

import loudml.vendor
from loudml.license import License


def generate(output, pub, priv, data):
    l = License()
    with open(pub, 'rb') as f:
        l.public_key = f.read()
    with open(priv, 'rb') as f:
        l.private_key = f.read()
    with open(data, 'rb') as f:
        l.payload_raw = f.read()

    l.save(output)


def inspect(license):
    l = License()
    l.load(license)

    print("File: " + license)
    print("Version: " + str(l.version))

    if not l.validate():
        print("Integrity: ERROR - invalid signature")
        exit(1)

    print("Integrity: OK")
    print("Contents: \n" + l.payload_raw.decode('ascii'))


def main(argv=None):
    parser = argparse.ArgumentParser(description="License management")
    parser.add_argument('-g', '--generate', help="generate license")
    parser.add_argument('--pub', help="public key")
    parser.add_argument('--priv', help="private key")
    parser.add_argument('--data', help="data file")

    parser.add_argument('-i', '--inspect', type=str,
                        help="inspect license")

    args = parser.parse_args()
    if args.generate:
        if args.priv is None:
            print("Private key required")
            exit(1)
        if args.pub is None:
            print("Public key required")
            exit(1)
        if args.data is None:
            print("Data file required")
            exit(1)
        generate(args.generate, args.pub, args.priv, args.data)

    if args.inspect:
        inspect(args.inspect)

#!/bin/sh
# Usage:
#    upload-artifacts.sh -d <distribution> *.deb
#
set -eu

if [[ -z "$APTLY_USER" ]]; then
    echo "Must provide APTLY_USER in environment" 1>&2
    exit 1
fi
if [[ -z "$APTLY_PASSWORD" ]]; then
    echo "Must provide APTLY_PASSWORD in environment" 1>&2
    exit 1
fi
if [[ -z "$GPG_PASSPHRASE" ]]; then
    echo "Must provide GPG_PASSPHRASE in environment" 1>&2
    exit 1
fi
aptly_user="$APTLY_USER"
aptly_password="$APTLY_PASSWORD"
gpg_passphrase="$GPG_PASSPHRASE"
aptly_api="https://artifacts.loudml.io"

while getopts ":d:" opt; do
  case ${opt} in
    d )
      dist=$OPTARG
      ;;
    \? )
      echo "Invalid option: $OPTARG" 1>&2
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      ;;
  esac
done
shift $((OPTIND -1))

packages=$*
folder=`mktemp -u tmp.XXXXXXXXXXXXXXX`

for file in $packages; do
    echo "Uploading $file..."
    curl -fsS -X POST -F "file=@$file" -u $aptly_user:$aptly_password ${aptly_api}/api/files/$folder
    echo
done

aptly_repository=${dist}
aptly_published=s3:amazon:releases_${dist}/${dist}

echo "Adding packages to $aptly_repository..."
curl -fsS -X POST -u $aptly_user:$aptly_password ${aptly_api}/api/repos/$aptly_repository/file/$folder
echo

echo "Updating published repo..."
curl -fsS -X PUT -H 'Content-Type: application/json' --data \
    "{\"Signing\": {\"Batch\": true, \"Passphrase\": \"$gpg_passphrase\"}}" \
    -u $aptly_user:$aptly_password ${aptly_api}/api/publish/$aptly_published

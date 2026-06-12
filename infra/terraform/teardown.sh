#!/bin/bash
set -e
export PATH="$HOME/bin:$PATH"

echo "=== Destroying dev infrastructure ==="
terraform workspace select dev
terraform destroy

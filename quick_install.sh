#!/bin/bash
# JIGYASA Quick Install - One command to set everything up

echo "🚀 JIGYASA Quick Installer"
echo "=========================="
echo ""

# Download and run the full installer
if command -v curl &> /dev/null; then
    curl -fsSL https://raw.githubusercontent.com/Sairamg18814/jigyasa/main/install_jigyasa.sh | bash
elif command -v wget &> /dev/null; then
    wget -qO- https://raw.githubusercontent.com/Sairamg18814/jigyasa/main/install_jigyasa.sh | bash
else
    echo "Error: curl or wget required"
    exit 1
fi
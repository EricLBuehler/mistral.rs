#!/usr/bin/env bash
# manage-mistralrs-service.sh
# Install or uninstall the mistralrs-qwen systemd service.
# Must be run as root or with sudo.

SERVICE_NAME="mistralrs-qwen"
SERVICE_FILE="${SERVICE_NAME}.service"
SYSTEMD_DIR="/etc/systemd/system"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$(id -u)" -ne 0 ]; then
    echo "Error: this script must be run as root (e.g. sudo $0 <install|uninstall>)"
    exit 1
fi

usage() {
    echo "Usage: sudo $0 <install|uninstall>"
    exit 1
}

install_service() {
    SRC="${SCRIPT_DIR}/${SERVICE_FILE}"
    DST="${SYSTEMD_DIR}/${SERVICE_FILE}"

    if [ ! -f "${SRC}" ]; then
        echo "Error: service file not found at ${SRC}"
        exit 1
    fi

    echo "Installing ${SERVICE_FILE} -> ${DST}"
    cp "${SRC}" "${DST}"
    chmod 644 "${DST}"

    echo "Reloading systemd daemon..."
    systemctl daemon-reload

    echo "Enabling service (auto-start on boot)..."
    systemctl enable "${SERVICE_NAME}"

    echo "Starting service..."
    systemctl start "${SERVICE_NAME}"

    echo ""
    echo "Done. Service status:"
    systemctl status "${SERVICE_NAME}" --no-pager
}

uninstall_service() {
    echo "Stopping service (if running)..."
    systemctl stop "${SERVICE_NAME}" 2>/dev/null || true

    echo "Disabling service..."
    systemctl disable "${SERVICE_NAME}" 2>/dev/null || true

    DST="${SYSTEMD_DIR}/${SERVICE_FILE}"
    if [ -f "${DST}" ]; then
        echo "Removing ${DST}"
        rm -f "${DST}"
    else
        echo "Service file not found at ${DST}, nothing to remove."
    fi

    echo "Reloading systemd daemon..."
    systemctl daemon-reload
    systemctl reset-failed 2>/dev/null || true

    echo "Done. ${SERVICE_NAME} has been uninstalled."
}

case "${1:-}" in
    install)
        install_service
        ;;
    uninstall)
        uninstall_service
        ;;
    *)
        usage
        ;;
esac

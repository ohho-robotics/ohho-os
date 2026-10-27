#!/usr/bin/env bash
# Prepend the virtualenv created by ros2-install.sh. Source this file.
if [[ -x /opt/ohho-ci/bin/python3 ]]; then
  case ":${PATH}:" in
    *:/opt/ohho-ci/bin:*) ;;
    *) export PATH="/opt/ohho-ci/bin:${PATH}" ;;
  esac
  hash -r
fi

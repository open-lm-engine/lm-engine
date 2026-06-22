REPO_BASE="20260221"
DOCKER_TAG="028a87180e7f9302636da4be3e06f0b4addb76e2"

enroot import --output \
    /data/users/hanguo/images/${REPO_BASE}-${DOCKER_TAG}.sqsh \
    docker://hanguo97/${REPO_BASE}:${DOCKER_TAG}

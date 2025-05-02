# multi2vec-clip-inference

The forked inference container for the clip modul

## Build Docker container

```
LOCAL_REPO="clip" \
  TEXT_MODEL_NAME="sentence-transformers/clip-ViT-B-32-multilingual-v1" \
  CLIP_MODEL_NAME="clip-ViT-B-32" \
  ./scripts/build.sh

```

## Run tests

```
LOCAL_REPO="clip" ./scripts/test.sh
```

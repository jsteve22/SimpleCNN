# SimpleCNN

```
docker build -t simplecnn .
docker run -it --name scnn --mount "type=bind,source=$PWD,target=/home/" simplecnn
```

To start the same container again use:
```
docker start -i scnn
```
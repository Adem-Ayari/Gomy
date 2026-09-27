to build
``` 
docker build -t currency-forecast .
```
then to run
```
docker run --rm \
           -v "$(pwd)/data:/app/data:ro" \
            -v "$(pwd)/output:/app/output" currency-forecast
```

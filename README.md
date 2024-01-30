# 🎁 JustRAIGS Challenge Pack 🎁
Thank you for hosting your challenge on Grand-Challenge.org. We appreciate it!

## Introduction

This Challenge Pack is a collection of challenge-tailored examples to help you on your 
way to hosting your JustRAIGS challenge.

Please note that this is a supplementary pack to the [documentation](https://grand-challenge.org/documentation/challenges/). 
If the documentation does not answer your question, feel free to reach out to us at 
[support@grand-challenge.org](mailto:support@grandchallenge.org).

The examples here are based on a POSIX system such as Linux or Macintosh OS.

## ⚙️ Data uploading ⚙️

Challenges pull their data for running phases from archives. 

For this challenge, two archives exist:
- [justraigs-development-phase-data](https://grand-challenge.org/archives/justraigs-development-phase-data/ )
- [justraigs-test-phase-data](https://grand-challenge.org/archives/justraigs-test-phase-data/)


To aid in helping to upload data to archives, we've created an example script
`./scripts/upload_to_justraigs-test-phase-data.py`. 
This generates a `archive_item_to_content_mapping.json` that helps in setting up the
evaluation method.

The script requires `Python`, `Pillow`, `tifffile`, and `gc-api` to be installed. Read more about setting up gc-api in 
the [grand-challenge.org documentation](https://grand-challenge.org/documentation/what-can-gc-api-be-used-for/).

As it is an example, it will likely require some tweaks before you can call:

```
$ python ./scripts/upload_to_justraigs-test-phase-data.py
```

You check the upload and processing here: https://grand-challenge.org/cases/uploads/

## ⚙️ Example algorithm ⚙️

An example algorithm container is provided via: `./example-algorithm`.

You can study it and run it by calling:

Navigate to the correct directory:
```
cd ./example-algorithm
```

Build the image and tag it with 'example-algorithm':
```
docker build --tag example-algorithm .
```

Remove any existing output, so we are sure the new output is from the run:
```
rm -f ./test/output/* 
```

Run the container using the image tag, mount the test/input and test/output directories:
```
    docker run --rm --gpus all --network none \
    --volume $(pwd)/test/input:/input \
    --volume $(pwd)/test/output:/output \
    example-algorithm
```

This should output something along the lines of:
```
=+==+==+==+==+==+==+==+==+==+=
Torch CUDA is available: True
        number of devices: 1
        current device: 0
        properties: _CudaDeviceProperties(name='NVIDIA GeForce GTX 1650 Ti', major=7, minor=5, total_memory=4095MB, multi_processor_count=16)
=+==+==+==+==+==+==+==+==+==+=
Input Files:
[PosixPath('/input/stacked-color-fundus-images/images/bd658ff4-d26f-4f08-ad6b-861e6f3e7c26.tiff'),
 PosixPath('/input/stacked-color-fundus-images/images/01479072-dd3f-4b7d-8813-dbee8aed2b08.mha')]
De-Stacked /tmp/tmpglwljyfs/image_1.jpg
De-Stacked /tmp/tmpglwljyfs/image_2.jpg
De-Stacked /tmp/tmpglwljyfs/image_3.jpg
Running inference on /tmp/tmpglwljyfs/image_1.jpg
Running inference on /tmp/tmpglwljyfs/image_2.jpg
Running inference on /tmp/tmpglwljyfs/image_3.jpg
Running inference on /tmp/tmpptq5kub5/image.jpg
```

You can prep it for upload using:
```
$  docker save example-algorithm | gzip -c > example-algorithm.tar.gz
```


## ⚙️ Example evaluation method ⚙️

An example evaluation method container is provided via: `./example-evaluation`. It does not, currently, do any 
sensible evaluation.

You can study it and run it by calling:

avigate to the correct directory:
```
cd ./example-evaluation
```

Build the image and tag it with 'example-algorithm':
```
docker build --tag example-evaluation .
```

Remove any existing output, so we are sure the new output is from the run:
```
rm -f ./test/output/* 
```

The intermediate folder is just to show the output of the csv
```
rm -f ./intermediate/* 
```

```
docker run --rm \
    --network none \
    --volume $(pwd)/intermediate:/tmp/intermediate \
    --volume $(pwd)/test/input:/input \
    --volume $(pwd)/test/output:/output \
    example-evaluation
```

You can prep it for upload using:
```
$  docker save example-evaluation | gzip -c > example-evaluation.tar.gz
```





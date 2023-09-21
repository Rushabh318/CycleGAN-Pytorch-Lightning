from minio import Minio
import argparse
import os
import pathlib

def fetch_data(source, dest, is_https=False):
    print("Started fetch data")  
    # Login to s3 bucket
    client = Minio(
        "{HOST}:{PORT}".format(HOST=os.environ["BUCKET_HOST"], PORT=os.environ['BUCKET_PORT']), 
        secure=is_https,
        access_key=os.environ['AWS_ACCESS_KEY_ID'], 
        secret_key=os.environ['AWS_SECRET_ACCESS_KEY'], 
        region=""
    )

    for obj in client.list_objects(os.environ['BUCKET_NAME'], prefix=None, recursive=True):
        if not obj._object_name.startswith(source):
            continue
        
        #print(f"Next object is {obj._object_name}.")
        if obj._object_name.split('/')[-1] != "":
            print(f"{obj._object_name} added to {dest}.")
            client.fget_object(os.environ['BUCKET_NAME'], obj._object_name, "/".join([dest, obj._object_name]))

def sync_data(source, dest, is_https=False):
    # print(f"host: {os.environ['BUCKET_HOST']}, access key: {os.environ['AWS_ACCESS_KEY_ID']}, secret key: {os.environ['AWS_SECRET_ACCESS_KEY']}, region: {os.environ['BUCKET_REGION']}")
    client =Minio(
        "{HOST}:{PORT}".format(HOST=os.environ["BUCKET_HOST"], PORT=os.environ["BUCKET_PORT"]), 
        secure=is_https, access_key=os.environ['AWS_ACCESS_KEY_ID'], 
        secret_key=os.environ['AWS_SECRET_ACCESS_KEY'], 
        region="")

    print("Sync s3")
    for root, dirs, files in os.walk(source):
        relative_path = os.path.relpath(root, source)
        for filename in files:
            result = client.fput_object(
                os.environ['BUCKET_NAME'], 
                "/".join([dest, relative_path, filename]), 
                os.path.join(source, filename), content_type="application/octet-stream")
            print(result.object_name)

def put_file(source, dest, is_https=False):
    client =Minio(
        "{HOST}:{PORT}".format(HOST=os.environ["BUCKET_HOST"], PORT=os.environ["BUCKET_PORT"]), 
        secure=is_https, access_key=os.environ['AWS_ACCESS_KEY_ID'], 
        secret_key=os.environ['AWS_SECRET_ACCESS_KEY'], 
        region="")
    
    print("Putting ", source, " to ", dest, " in ", os.environ['BUCKET_NAME'] )
    client.fput_object(os.environ['BUCKET_NAME'], dest, source)

def push_data(source, dest, is_https=False):
    # print(f"host: {os.environ['BUCKET_HOST']}, access key: {os.environ['AWS_ACCESS_KEY_ID']}, secret key: {os.environ['AWS_SECRET_ACCESS_KEY']}, region: {os.environ['BUCKET_REGION']}")
    client =Minio("{HOST}:{PORT}".format(HOST=os.environ["BUCKET_HOST"], 
                                         PORT=os.environ["BUCKET_PORT"]), 
                  secure=is_https, 
                  access_key=os.environ['AWS_ACCESS_KEY_ID'],
                  secret_key=os.environ['AWS_SECRET_ACCESS_KEY'], 
                  region="")

    print("Sync s3")
    source = pathlib.Path(source)
    for root, dirs, files in os.walk(source):
        relative_path = os.path.relpath(root, source)
        for filename in files:
            print(pathlib.Path.cwd())
            file_dest = pathlib.Path(dest) / pathlib.Path(relative_path) / filename
            file_source = pathlib.Path(root) / filename
                    
            result = client.fput_object(os.environ['BUCKET_NAME'], 
                                        file_dest.as_posix(), 
                                        file_source.as_posix(), 
                                        content_type="application/octet-stream")
            print(result.object_name)

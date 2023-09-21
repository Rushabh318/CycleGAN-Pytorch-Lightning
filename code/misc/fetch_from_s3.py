from utils.s3_tools import fetch_data

if __name__ == '__main__':
    remote_dir = 'lr/data/ct/'
    source = remote_dir + 'source/'
    labels = remote_dir + 'labels/'
    
    fetch_data(source, '../data/source')
    fetch_data(labels, '../data/labels')
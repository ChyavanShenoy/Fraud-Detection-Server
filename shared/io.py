import requests
import bs4
import os
import glob


def get_files(url, dest):
    download_dir = dest
    r = requests.get(url)
    data = bs4.BeautifulSoup(r.content, 'html.parser')
    for link in data.find_all('a'):
        r = requests.get(url + link.get('href'))
        print('Downloading file: ', link.get('href'))
        print(r.status_code)
        if r.status_code == 200:
            with open(download_dir + '/' + link.get('href'), 'wb') as f:
                f.write(r.content)
    return


def delete_files(folder):
    for data in glob.glob(folder+'/*.*'):
        os.remove(data)

# get_files('http://192.168.29.226:9000/', './Temp')
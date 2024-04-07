import gdown

url = 'https://drive.google.com/uc?id=1U1pKU8d5sVhsQ3-_dCuWx_W6jStdRq8j'
output = 'best.pkl' 

gdown.download(url, output, quiet=False)

# The following code will only execute
# successfully when compression is complete

import kagglehub

kagglehub.login()
# Download latest version
path = kagglehub.competition_download('icaif-24-finance-rag-challenge')

print("Path to competition files:", path)
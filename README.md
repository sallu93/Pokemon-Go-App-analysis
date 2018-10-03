# Pokemon Go! App analysis using Web Analytics and Machine learning

In this project, we tried to understand the success of the game app Pokemon Go, using the programming language Python.

## Project Steps

- Web scraping using BeautifulSoup was done to extract useful values from the raw HTML files. 

- Next, the data was organized by creating "Dictionaries" and converting them into "Pandas dataframe" to facilitate data exploration. The data was organized using "datetime" as we had time series data. The dataframe was saved in three formats (JSON, CSV and  XLSX).

- The data was explored using the previously created Pandas dataframe. Methods like scatter_matrix(), corrcoeff() and matplotlib were used for the purpose.

- Finally, Machine learning models were built on the success of the app using the "number of ratings" values from the data collected. Two Regression models (One for iOS and the other for Android) were built using "sklearn".


## Data description:

For the project, the app pages of Pokemon Go! were downloaded from Google Play Store and Apple App Store from July 21 2016 to October 31 2016:

• https://play.google.com/store/apps/details?id=com.nianticlabs.pokemongo&hl=en

• https://itunes.apple.com/us/app/pok%C3%A9mongo/id1094591345?mt=8

The webpages were downloaded every ten minutes. This means that there were 144 (=24x6) HTML files for a given day and a given platform. There were a total of 103 date folders in a master folder called "data". Each date folder contained HTML files downloaded in the specified date. Each HTML file name was formatted as “HH_MM_pokemon_PLATFORM.html”, where HH is hour, MM is minute, and PLATFORM is either “android” or “ios”. Note that due to intermittent connection errors, some HTML files were not properly downloaded.

## Other Info

- Other contributor: Tapan Patel

- Programming language used: Python





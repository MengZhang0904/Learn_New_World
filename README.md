# Learn_New_World
This is my individual data science project at Insight.

EdTech is rising. Parents are always willing to invest on their children.
I created this end-to-end web app (www.childlearn.info) to provide personalized word learning info and recommendations to parents for their children.

## What happened on the users' end?
When parents enter the age of their kids and words they just learned, my app will tell parents how hard the word is based their kids' age and the learning curve of the word.

More importantly, it also provides other words kids at this age may also know and a list of words they are ready to learn in the next month.

## What happened under the hood?
Under the hood, I made use of children's vocabulary data on a standard test from a language development lab at Stanford.

The difficulty and the learning curve are based on statistics of the data.

The recommendation of similar words and learnable words in next two months came from calculating distances between words in vector space.

I trained my own word2vec embeddings for vocabulary of each age group. The recommended words are words that are developmentally close to the target words.

I launched my web app on AWS and surveyed around my friends and colleagues who have kids, they all like it.

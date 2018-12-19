---
layout: post
title:  "Tweet-to-tweet learning"
---

_Note: this post is still work in progress_

Natural language processing is, in my opinion, one of the most interesting application of
machine learning. Recurrent neural networks are by far the most successful models in this
context, with LSTM and GRU networks being the top choice.  

One can find several excellent tutorials and blog posts on how to train a recurrent neural
network on a text corpus. I would definitely recommend reading
"[The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness){:target="_blank"}"
from A. Kaparthy for a review of cool applications of RNNs.  

Deep learning frameworks such as
[Keras](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html){:target="_blank"}
and [PyTorch](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html){:target="_blank"}
also provide extensive documentation on how to build such a model.  

So is this yet another blog post about training RNNs? Obviously the answer is 'No' (as for
all rhetorical questions). What did in this experiment was to go though the whole process
of training a words sequence generator from end-to-end, i.e. from scraping the web to collect the data,
to setting up an application that exploits the generative model.  

In practice I built a tweets dataset streaming posts related to machine learning, trained
a recurrent neural network to learn the distribution of these tweets, and finally used the
model to build a tweet autocompletion application.

The inpatients can see the final artefact by clicking on the image below. The code that I
used is available in [this](https://github.com/musella/tweetter_crawl) repository. Below I
describe in details all the steps of the project.

{: style="max-width:60%" }
[![twitter_rnn_webapp.png](/assets/twitter_rnn_webapp.png)](http://musella.pythonanywhere.com/){:target="_blank"}

## Collecting and cleaning the data


### Streaming tweets

[tweet_streamer.py](https://github.com/musella/twitter_crawl/blob/master/scripts/tweet_streamer.py)

{% highlight python %}
l = StdOutListener()
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

stream = Stream(auth, l,tweet_mode="extended")
stream.filter(track=['NeurIPS2018','MachineLearning'])

{% endhighlight %}

### Preprocess and clean

[preprocess.ipynb](https://github.com/musella/twitter_crawl/blob/master/notebooks/preprocess.ipynb)

## Words representation

The (GloVe)[https://nlp.stanford.edu/projects/glove/] authors trained and published a
words embedding based on tweets data.  

The GloVe embedding does not know about the specific hashtags used in our dataset, so we
train a dedicated embedding for hashtags.

{: style="max-width:60%" }
[![twitter_rnn_hash_embedding.png](/assets/twitter_rnn_hash_embedding.png)](/assets/twitter_rnn_hash_embedding.png){:target="_blank"}

The joined representation is constructed by taking the product of the words and hashtags embedding space.

{: style="max-width:60%" }
[![twitter_rnn_sentences.png](/assets/twitter_rnn_sentences.png)](/assets/twitter_rnn_sentences.png){:target="_blank"}

## Learning the sequence distribution

$\log( ~ p(\vec y) ~ )  = \sum_{i=1}^N \log \left( ~p(y_i \vert y_{i-1},...,y_1) ~\right )$

## Generating sequences with beam search

![twitter_rnn_beam_search.png](/assets/twitter_rnn_beam_search.png)

## Wrapping it all up in web application using Flask

[app.py](https://github.com/musella/twitter_crawl/blob/master/flask_app/app.py)

{% highlight python %}
def run_beam_search(request,horizon):
        if request.method == 'POST':
                print("POST request")
        else:
                print("GET request")
                
        text = request.json['input']

        print("Horizon", horizon)
        print("Text", text)
        suggestions = beam_search.predict(text,nsuggestions,horizon)
                
        return jsonify(suggestions=suggestions)

        
@app.route('/autocomplete', methods=['GET', 'POST'])
def autocomplete():
        what = request.json["what"]
        if what == "get_next_word":
                return run_beam_search(request,1)
        elif what == 'get_sentence':
return run_beam_search(request,50)

{% endhighlight %}

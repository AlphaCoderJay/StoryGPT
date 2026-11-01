# StoryGPT

# TinyStories Transformer

A GPT-style language model trained from scratch on the TinyStories dataset.

## About

This project started with a simple goal: I wanted to build and train my own language model, rather than just use a pretrained one.

I implemented a decoder-only Transformer and trained it on TinyStories, a dataset of short stories. The idea was to take what I'd been learning about Transformers and put it into practice by going through the entire process myself, from preparing the dataset to training the model and generating text.

## The Goal

The main goal wasn't to build the next ChatGPT. It was to understand what actually goes into training a language model.

I wanted to get comfortable with the fundamentals:
- Building a GPT-style Transformer architecture.
- Understanding self-attention and how tokens interact with one another.
- Preparing and processing text for training.
- Training with next-token prediction.
- Experimenting with hyperparameters and observing how they affect training.
- Generating text using the trained model.

## The Dataset

I used **TinyStories**, a dataset consisting of short, relatively simple stories.

It provides a manageable dataset for training smaller language models without requiring the compute normally associated with large-scale pretraining.

## Sample Generations

Here are a few of the better generations from my trained model.

### Sample 1

> Tim found a mysterious box under his bed. He opened the box and found a big, colorful hat inside. Tim put the hat on his head and went to show his friends.

Tim's friends came over to play. They all took turns wearing the hat. Tim and his friends had a fun day playing with the mysterious box. They laughed and played until it was time for Tim to go home.

### Sample 2

> One morning, Lucy woke up and discovered something stange. A small green creature was wearing a blue shirt. The creature looked at the girl and said, "It looks like it's me". Lucy's eyes lit up and she asked, "What's your name?" The creature said, "My name is Bob. I'm playing with your ball. Do you want to come play with me?" Lucy was so excited and she said, "Yes please! Let's play together!"
                So Lucy and Bob played with the ball near the woods. They had a lot of fun. They both became good friends and promised to always remember the little creature. And they lived happily ever after.
### Sample 3
> It was a beautiful sunny day when something unexpected happened. A small bird flew down and landed on the tree, waiting to see the big round ball.
The bird looked up and saw a little girl. The little girl was sad because she had no toys to play with. The bird flew down and said, "You can play with me!" The little girl was happy and they played together all day. From that day on, the little girl and the bird became best friends.

## What's Next?

This project is one of the steps in my journey towards understanding how modern LLMs are built.

The next stage is to experiment with more modern Transformer architectures, introducing techniques such as RoPE, RMSNorm, SwiGLU and Grouped Query Attention, and comparing them against the standard Transformer.

This repository focuses on the original Transformer implementation. The modern architecture experiments are being developed separately.

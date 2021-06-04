---
layout: post
title: 'Introduction to transfer learning'
date: '2021-06-01 15:04'
excerpt: >-
  Can deep neural networks acquire knowledge from a fellow neural network ? Let's find out.
comments: true
tags: [june_2021, deep_learning, transfer_learning]
---

If we think intuitively, it is not realistic and practical for us to learn everything from scratch. We always try to solve a new task based on the knowledge obtained from past experiences. For example, we may find that learning to recognize apples might help to identify pears or learning a programming language, say C++, can facilitate learning some other language, say Python, as the basic programming fundamentals remain the same.

Transfer learning is an area of deep learning inspired by the same idea. We try to use the "knowledge" acquired in one task to do some other related task. This is mainly motivated by the lack of labeled data across domains. A simple example would be the ImageNet dataset, which has millions of images of different categories. However, getting such a dataset for every domain is challenging. Besides, most deep learning models are very specialized to a particular domain or even a specific task. They have high accuracy and beat all benchmarks, but only on particular datasets, and end up suffering a significant loss in performance when used in a new task that might still be similar to the one it was trained on.

## Why transfer learning?


![image1](/images/transfer_learning/paradigm_difference_machine_learning.png) | ![image2](/images/transfer_learning/paradigm_difference_transfer_learning.png)


In the classic supervised learning scenario of machine learning, we use labeled data to train a model on a specific task and domain $A$. Say, we are considering the problem of sentiment classification, where the task is to automatically classify the reviews on a product, such as a brand of camera, into positive and negative views. Now the distribution of review data for different types of products can be dissimilar, so to maintain good classification performance on all the products, we need to collect a large amount of labeled data for every product.

However, this kind of data labeling is very expensive to do. To reduce this effort, we may want to train a few classification networks and then adapt the learning for some other products. This is where transfer learning kicks in. We are essentially trying to use some trained network to make predictions on a related task that is different from what the network was initially trained on.

## What is transfer learning
To go further we need to familiarize ourselves with some notations and definitions.

* **Domain** : Given a specific dataset $X = \set{x_1, … , x_n} ∈ X$ , where $X$ denotes the feature space, and a marginal probability distribution on the dataset $P(X)$. A domain can be defined as $D(X, P(X))$. For example, if our learning task is document classification, and each word is taken as a binary feature, then $X$ is the space of all word vectors, $x_i$ is the $i$th word vector corresponding to some documents, and $X$ is a particular learning sample. In general, if two domains are different, then they may have different feature spaces or different marginal probability distribution.

* **Task** : For a specific domain, $D = {X, P(X)}$, a task consists of two components: a label space $Y$ and an predictive function $f(·)$ (denoted by $T = {Y, f(·)}$), which is learned from the training data, which consist of pairs $ \set{x_i, y_i}$, where $x_i ∈ X$ and $y_i ∈ Y$. The predictive function $f(.)$ can be seen as a conditional distribution $P(Y\|X)$. In our document classification example, Y is the set of all labels, which is True, False for a binary classification task.

Now that we understand what a domain and a task is we take the formal defination of transfer learning from a paper written by Pan and Yang<sup>[1](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf)</sup>

**Transfer Learning**: Given a source domain $D_s$ and its corresponding task $T_s$, where the learned function $f_s$ can be interpreted as some knowledge obtained in $D_s$ and $T_s$. Our goal is to get the target predictive function $f_t$ for target task $T_t$ with target domain $D_t$. Transfer learning aims to help improve the performance of $f_t$ by utilizing the knowledge $f_s$, where $D_s \neq D_t$ or $T_s \neq T_t$.
In short, transfer learning can be simply denoted as
$\begin{align}
D_s, T_s \rightarrow D_t, T_t
\end{align}$

## Different scenarios in transfer learning

Based on $D_s \neq D_t$ or $T_s \neq T_t$. , we can have three scenarios when applying transfer learning.

![scenarios in transfer learning](/images/transfer_learning/different_scenarios_of_transfer_learning.png)

When $D_s = D_t$ and $T_s = T_t$, the problem becomes a traditional deep learning task. In such case, a dataset is usually divided into a training dataset $D_s$ and a test training dataset $D_t$, then we can train a neural network $F$ on $D_s$ and apply the pre-trained model $F$ to $D$

When the domains are same i.e. $D_s$ = $D_t$ and $T_s \neq T_t$, the problem becomes a multi-task learning problem i.e. As the source domain and the target domain have the same feature space, we can utilize one giant neural network to solve different types of tasks simultaneously. If the tasks are different, then either
1. The label spaces between the domains are different, i.e. $Y_s \neq Y_t$ or
2. The conditional probability distributions between the domains are different; i.e. $P(Y_s\|X_s) \neq P(Y_t\|X_t)$

In our document classification example, case 1 corresponds to the situation where the source domain has binary document classes, whereas the target domain has ten classes to classify the documents. Case 2 corresponds to the situation where the source and target documents are very unbalanced in terms of the user-defined classes

Finally, If the domains are different and $T_s$ is similar to $T_t$, we use deep domain adaptation techniques. Where the goal of domain adaptation is to learn $f_t$ from $f_s$ when we change domains. Clearly when $D_s \neq D_t$ either
1. The feature spaces between the domains are different, i.e. $X_s \neq X_t$ , or
2. The feature spaces between the domains are the same but the marginal probability distributions between domain data are different i.e. $P(X_s) \neq P(X_t)$

As an example, in our document classification example, case 1 corresponds to when the two sets of documents are described in different languages, and case 2 may correspond to when the source domain documents and the target domain documents focus on different topics

As a side note, when $D_s \neq D_t$ and $T_s \neq T_t$, transfer learning becomes complicated. This is technically called **negative learning**. Suppose the data in source domain $D_s$ is very different from that in target domain $D_t$. In that case, brute force transfer may hurt the performance of predictive function $F_t$, not to mention the scenario when source task $T_s$ and target task $T_t$ are also different. This is still an area of open research. You can read [this](https://arxiv.org/pdf/2009.00909.pdf) survey paper

## Ways to use transfer learning

Now that we understand transfer learning, we are faced with the question how to apply transfer learning. The essential idea is to use a pre-trained network and then tweak it to serve our purpose. Coming from traditional machine learning, if we wanted to use our network, we'll load it in memory and give it an example to make predictions on. The network then feed forwards this example to all the layers and gives us the prediction. Usually in deep neural networks just before the output, a fully connected layer utilizes whatever the netwrok has learned into making its final prediction. So if we were to use this network's "knowledge," we have several options to extract it.

For simplicity's take lets take the example of a ConvNet trained on the ImageNet dataset (Alexnet for example) and understand how we can use this to transfer the knowledge.

![ConvNet in different scenarios](/images/transfer_learning/different_CNN_Models.png)
*Vanilla CNN ; CNN as a feature extractor ; CNN with fine tuning and feature extraction*

- As a fixed feature extractor: Here the idea is to take a ConvNet pretrained on ImageNet, remove the last fully-connected layer (the layer having its outputs as the 1000 class scores for a different task), then treat the rest of the ConvNet as a fixed feature extractor for the new dataset. In essence, we are only using this network to get the final feature map of the input. In an AlexNet, this would compute a 4096-D vector for every image that contains the activations of the hidden layer immediately before the classifier. We call these features CNN codes. It is important for performance that these codes are ReLUd _(yeah that's a word)_ i.e. thresholded at zero if they were also thresholded during the training of the ConvNet(as is usually the case). Once you extract the 4096-D codes for all images, train a linear classifier (e.g. Linear SVM or Softmax classifier) for the new dataset.

- Fine-tuning the ConvNet: It will not always we sufficient for us to just use the network as it is and make do with letting go of the final layer. So, the second strategy is to not only replace and retrain the classifier on top of the ConvNet on the new dataset, but to also fine-tune the weights of the pretrained network by continuing the backpropagation. It is possible to fine-tune all the layers of the ConvNet, or it's possible to keep some of the earlier layers fixed (due to overfitting concerns) and only fine-tune some higher-level portion of the network. This is motivated by the observation that the earlier features of a ConvNet contain more generic features (e.g. edge detectors or color blob detectors) that should be useful to many tasks, but later layers of the ConvNet becomes progressively more specific to the details of the classes contained in the original dataset. In the case of ImageNet, which contains many dog breeds, a significant portion of the representational power of the ConvNet may be devoted to features that are specific to differentiating between dog breeds.

- Pretrained models. Since modern ConvNets take 2-3 weeks to train across multiple GPUs on ImageNet, it is common to see people release their final ConvNet checkpoints for the benefit of others who can use the networks for fine-tuning. For example, the Caffe library has a [Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) where people share their network weights.

Now that we know how domain and task can differ and how we intend to use the pretrained network, here are some ideas so as to how to use transfer learning in a given context.

![sample flow chart type thingy](/images/transfer_learning/dataset_relation.png)

- New dataset is small and similar to original dataset: Training neural networks on small datasets can lead to overfitting, so fine-tuning the network is not a good idea. But since the datasets are similar gigher level features in the ConvNet will be relevant. So training a linear classifier on CNN nodes is a good idea in this case.

- New dataset is large and similar to the original dataset: This is the best case scenario. Since you have a lot of data you can fine-tune the network with ease without worrying about overfitting. The best-idea, hence, is to fine-tune the network

- New dataset is small but very different from the original dataset: This is a challanging problem of domain adaptation. As explained earlier, since the data is small, it is likely best to only train a linear classifier. Since the dataset is very different, it might not be best to train the classifier form the top of the network, which contains more dataset-specific features. Instead, it might work better to train the SVM classifier from activations somewhere earlier in the network. Using a prior checkpoint in the pre-trained network will work best in this case. However, there are chances of negative transfer.

- New dataset is large and very different from the original dataset: Since the dataset is very large, we may expect that we can afford to train a ConvNet from scratch. However, in practice it is very often still beneficial to initialize with weights from a pretrained model. In this case, we would have enough data and confidence to fine-tune through the entire network.

## Working through an example

As with everything, its good to practice a few problems when you learn a new technique. I found [this](https://www.kaggle.com/c/dogs-vs-cats/data) dataset at Kaggle recommended in a blog post for implementing transfer learning using VCG-16 model. I am currently working through it.

### Resources

1. [A Survey on Transfer Learning](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf)
2. [Transfer Learning - Machine Learning's Next Frontier](https://ruder.io/transfer-learning/)
3. [How transferable are features in deep neural networks?](https://arxiv.org/pdf/1411.1792.pdf)
4. [CNN for Visual Recognition](https://cs231n.github.io/transfer-learning/)
5. [Survey of Deep Transfer Learning](https://arxiv.org/pdf/1808.01974.pdf)
6. [A Comprehensive Hands-on Guide to Transfer Learning with Real-World Applications in Deep Learning](https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a)

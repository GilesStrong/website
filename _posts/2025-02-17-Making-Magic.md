---
title: "Making Magic with LLMs"
description: "Building 'Magic: The Gathering' decks using LLM chains. AKA non-exhaustive, qualitative combinatorial optimisation."
layout: post
toc: true
comments: true
image: images/posts/2025-02-08/header.png
hide: false
search_exclude: false
categories: [AI, LLMs, NLP, fun]
---

<img src="{{ site.baseurl }}/images/posts/2025-02-08/header.png" style="width:auto;height:auto" />

Magic: The Gathering (MTG, Magic) is world-famous a game I have been playing since childhood. At its core, it is a collectible card game where players build decks of cards representing magical spells, creatures, and artifacts. The game is relatively easy to get started, but has a high skill ceiling due to the complexity of the interactions between cards. Because of this, the fun in playing comes not just from the game itself, but also from the deck-building process. New cards are continually released, and other cards "rotate out" of what is currently legal to play in the standard format. This means that the metagame is always changing, and players need to adapt their decks to stay competitive.

Recently, I began to get back into playing Magic, through their online platform called [Arena](https://magic.wizards.com/en/mtgarena). However, given the vast number of cards available, and my limited time to play, I wanted to find a way to quickly build decks that were interesting and fun to play. At the same time, I needed a project to help expand my skills in constructing LLM-based systems, and thought the two could be combined. In this post, I will describe  the system I constructed to generate Magic decks using LLMs, and some of the challenges I encountered along the way.

If you want to try out the system I built, it is available here: [DeepMTG](https://github.com/GilesStrong/deep_mtg)

## The fundamentals

A typical deck consists of 60 cards. The cards are divided into two main categories: lands and spells. Lands are able to produce "mana" -- the resources that players use to cast spells -- and spells are the cards that represent the magical effects that players can cast. The spells are further divided into different types, such as creatures, enchantments, artifacts, sorceries, and instants. Each card has a "colour" associated with it, which determines the mana resources needed to cast it. There are five colours in Magic: white, blue, black, red, and green. Each colour has its own strengths and weaknesses, and different strategies can be built around them.

When playing, cards are drawn every turn, randomly from the deck. This means that the deck needs to be balanced in terms of the number of lands and spells, and the distribution of colours. The deck also needs to have a good "curve", meaning that the cards should be of different mana costs, so that the player can cast something every turn. Finally, the deck should have a coherent strategy, so that the cards work well together in order; since the number of copies of a given card is limited to four, decks cannot rely on drawing a given card, and instead must include redundancy and a sufficient range to ensure that games are not lost early.

The general aim is to defeat the opponent, by reducing their life total to zero. This can be done by attacking with creatures, or by casting spells that deal damage directly to the opponent. However, there are also other ways to win, such as by making the opponent run out of cards to draw, or by playing a cards that have a special win conditions.

Spells are necessarily only offensive, they can also be used to protect your own creatures, to disrupt the opponent's strategy by blocking their attacking creatures, making them discard cards, etc., or utility spells that can draw you more cards, or search for specific cards in your deck.

One of the key features of Magic, is the "stack". When a spell is cast, it goes on the stack, and the opponent has a chance to respond by casting their own spells. This means that the order in which spells are cast can be very important, leading to complex interactions between cards, and even to infinite loops.

## The aim

I wanted to build a system that can take in a simple prompt in plain English, and generate a decklist that can be played in the standard format. The system should be able to generate decks that are fun to play, and that have a coherent strategy. The system should also be able to take prototype decks as a starting point and build out from their.

From an optimisation point of view, the problem is to maximise an unknown, and hard-to-quantify objective function defined over a 60-dimensional, categorical space. This is a difficult problem, since even if the objective function were easy to quantify, and the categories could be somewhat ordered, the surface is still likely to be extremely problematic, with many local, degenerate minima, that are misaligned with the axes of the space, due to the inherent reliance of performance on the interactions between cards.

An exhaustive search is also out of the question, since there are about 3500 cards in the standard format, and the number of possible decks is astronomical. Based on this, we cannot attempt to use black-box optimisation methods, and instead need an optimiser that can take advantage of understanding the game and the impact of its choices.

### Baseline

As a baseline comparison, I tried using ChatGPT to generate decks, however whilst it was quick to do so, it was quite an iterative process that wasn't always successful. Below is an example:

I begin by asking for a standard-legal deck with a clear theme:

<img src="{{ site.baseurl }}/images/posts/2025-02-08/IMG_1729.JPEG" style="width:auto;height:512px" />

The model responds with a decklist that fits to my theme, and is able to explain its choice, and strategy:

<img src="{{ site.baseurl }}/images/posts/2025-02-08/IMG_1730.JPEG" style="width:auto;height:512px" />

However, it included cards that are not currently allowed in the standard format:

<img src="{{ site.baseurl }}/images/posts/2025-02-08/IMG_1731.JPEG" style="width:auto;height:512px" />

It was, however, able to use web searches to find which cards are currently legal:

<img src="{{ site.baseurl }}/images/posts/2025-02-08/IMG_1733.JPEG" style="width:auto;height:512px" />

Even after revision, however, the deck still had incompatible cards, e.g. one of the centre pieces of the deck was a card that could not be played due to the deck having no sources of the required mana:

<img src="{{ site.baseurl }}/images/posts/2025-02-08/IMG_1734.JPEG" style="width:auto;height:512px" />

Additionally, the deck included cards that were not compatible with the strategy. In this example, the deck was supposed to be a "control" deck, but the model included cards that could not be used well other cards in the deck. After I pointed this out, it eventually removed them.

<img src="{{ site.baseurl }}/images/posts/2025-02-08/IMG_1735.JPEG" style="width:auto;height:512px" />

To ChatGPT's credit, it was able to generate a deck quickly, was able to explain its choices, and adapt to feedback. However, this was a very iterative process that required me to analyse the deck at each step.

## Design thought-process

I need to restrict the system to only be able to include cards that a) actually exist, and b) are legal in the standard format. This means that I need to have a database of all the standard-legal cards in the game, and a way to check if they are legal. However, there are currently about 3500 cards in standard at the moment, and feeding all of them into the system would be impractical: likely, the model will need to think through additions to the deck, in order to build strategies that are coherent, and including all the possible cards might add too much noise. Instead, I should give the system the ability to search for possible cards to add to the deck. By ensuring cards are only added from the database, I can then ensure that the deck is legal.

I also want to ensure that the deck builds a coherent strategy, and sticks to cards colour that can either fit in with the mana base; or are worth the effort to include the necessary mana sources. This means that the system needs to be able to think through the strategy of the deck, and ensure that the cards work well together.

Another consideration is that the system should know the rules of the game. Whilst these change occasionally, the basic rules of the game have remained the same for a long time, and luckily, it seems that the rules were included in the training data for many of the well-known LLMs. This means that the system should be able to understand the basic rules of the game. Whilst I did build a subsystem to allow the latest rules to be queried, in the end, this was not necessary.

One way to achieve both of the remaining concerns, then, is through the iterative addition of cards to a growing deck, much like the process I would go through myself, when building a deck. The system can start with a simple prompt, and then add cards to the deck, one at a time, until it reaches the desired deck size. At each step, the system can evaluate the deck, and decide which card to add next, based on the current state of the deck.

The basic loop would look something like this:

1. Analyse the state of the deck
2. Decide which card to add next
3. Search for that card in the database
4. Add the card to the deck
5. Repeat until the deck is complete

The basic data structure for the deck will be a list of cards, a prompt that describes the deck, and the current analysis of the deck.

## Current analysis of the deck

This is one of the more simple parts to write: given the current list of cards, we want to know what the current strengths and weaknesses of the deck are, and what strategies currently exist, or could be reinforced.

Do do this, I use an LLM with a prompt advising it that I will pass in a prototype deck, and that I want it to analyse the deck and provide a summary of its strengths and weaknesses:

<img src="{{ site.baseurl }}/images/posts/2025-02-08/analysis.png" style="width:auto;height:auto" />


## Card selection

Next, we need to choose which card to add next, ideally to account for the analysis of the deck. The difficulty is that the system does not know what cards currently exist in the database, until they are added to the deck. This means that the card-addition subsystem needs to be able to search for cards that fit the current state of the deck, and that can be added to the deck.

### Card request

To begin with, I instruct an LLM to provide a description of the card that should be added, using general terms that indicate the type of card that should be added, and what it should achieve:

<img src="{{ site.baseurl }}/images/posts/2025-02-08/request.png" style="width:auto;height:auto" />


### Card search

The request from the previous step can be treated as a search query, and used to find the most closely matching card in the database. To do this, I convert the search query into a vector, using an embedding model. The *similarity* between the embedded search query, and the existing cards in the database, can be calculated. 

Embedding models are used to convert variable-length text into fixed-length vectors, in which similar *concepts* are close together in the vector space. This means that even if the search query does not exactly match existing text in the database, we can still find suitable results.

This does, however, mean that we need to ensure that the cards are also able to be suitably encoded. Currently they exist in a json format. Whilst this may be excellent for encoding data, it does not really capture information about strategy or the card's effects in practice. This means that it may be difficult to match the plain-text search query to the cards in the database.

Instead, I decided to use an LLM to summarise each card in isolation and then include this summary in the embedding of the card. This means that the search query can be matched to the summary of the card, and the card can be added to the deck if it is a good match.

The way in which the summaries are generated is one of the key parts of the system, since it is the boundary between the card database (i.e. what actually exists), and the rest of the deck-design system. The summaries need to be informative enough to allow the system to make good decisions about which cards to add, but not so detailed that they are difficult to match to the search query.

#### Summary generation

I spent a long time comparing many models by hand, in order to find the best one for this task. To do this, I generated summaries for 10 cards using each model, and then graded them based on how well the summary captured information on:

- Cards cost and colour, 1 point
- Car role in the game, 1 point
- Type and subtype of card, 1 point
- The card's effects, 1 point
- What strengths and weaknesses the card has, 2 points
- What kind of decks the card is good in, 1 point
- And finally, how well the summary was written, 2 points

For any mistake in the summary, I would deduct 1 point. 

This resulted in the following scores:

| Model | Score |
|-------|-------|
| GPT-4o| 83 / 90 |
| GPT-4o-mini| 76 / 90 |
| deepseek-r1:14b | 75 / 90 |
| phi4:14b | 73 / 90 |
| mixtral:8x7b|65 / 90 |
| qwen2.5:32b | 56 / 90 |
| llama3.1:8b | 38 / 90 |
| mistral-small:24b| Ungraded|
| solar-pro:22b| Ungraded|
| qwq:32b| Ungraded|

With the exception of GPT, the rest of the models were self-hosted and so were much smaller in size, but were cheaper to run. The GPT model was the most expensive to run, but also provided the best summaries. Another concern is the time taken to generate a summary. R1 was very competitive with GPT, but was relatively slow to run, at about 7.8s per card. Phi4 was almost as good, but much quicker to run at around 3s per card.

Ungraded models are ones that I began testing, but did not complete due to them clearly not being as good as the others.

As a comparison, let's look at example summaries for one of the cards:

llama3.1:8b  7/9:
```
Banner of Kinship is a strong artifact that enhances your creatures based on their type
It has an expensive mana cost, but it's worth it as it can boost multiple creatures at once
To use its ability, you'll need to choose a creature type as it enters the battlefield
Once chosen, all creatures of that type under your control will gain +1/+1 for each fellowship counter on Banner of Kinship, making it an excellent tool for tribal decks.
```

Here llama's summary contains no information about the card cost, nor its weaknesses, but otherwise is a good summary.

phi4:14b 8/9:
```
**Banner of Kinship**  - **Mana Cost:** {5} - Moderate cost, typically requiring a mix of colored and colorless mana
   - **Role & Function:** This artifact is a support piece designed to boost the power and toughness of creatures you control
It uses fellowship counters as a mechanism for enhancement
 - **Mechanics:**   - Upon entering the battlefield, choose a creature type
   - The artifact receives a fellowship counter for each creature you control of that chosen type
  - Each chosen-type creature gets a +1/+1 bonus for every fellowship counter on the artifact
 - **Strengths:**   - Can provide significant power and toughness boosts to creatures if you have many of the chosen type
  - The choice of creature type allows flexibility based on your current board state or deck strategy
 - **Weaknesses:**   - Limited to boosting only one creature type at a time, which can be restrictive in diverse decks
  - Dependent on having multiple creatures of the chosen type for optimal effectiveness
  Overall, Banner of Kinship is best utilized in decks with a focus on specific creature types, allowing you to maximize its potential by creating synergy among your creatures.
```

Phi4's summary is very detailed, and nicely breaks down the information about the card. I deducted a point for suggesting that a mix of coloured and colourless mana is required, which is not necessarily true.

mixtral:8x7b 9/9:
```
5 mana colorless Artifact - Banner of Kinship  Banner of Kinship is a flexible card that can boost the power and toughness of your creatures
After it enters the battlefield, you choose a creature type, and it receives a fellowship counter for each of those creatures you control
Your chosen creatures get +1/+1 for each fellowship counter on Banner of Kinship, making them stronger as you have more of them in play
 Strengths: - Can increase the power and toughness of multiple creatures at once
- The cost is colorless, so it can fit into any deck
 Weaknesses: - Requires careful planning to maximize its potential
- The effect depends on having creatures of a specific type in play
 Banner of Kinship can be a strong card if used correctly, but it requires a well-built and focused creature-based deck for maximum impact.
```

Mixtral's summary is also very good and perfectly captures the information about the card.

solar-pro:22b 3/9:
```
Banner of Kinship is an Artifact card that costs five mana to play
It has no keywords and is colorless
When it enters the battlefield, you can choose a creature type and place a fellowship counter on it for each creature you control with that chosen type
This artifact provides bonuses to all creatures of the chosen type under your control by giving them +1/+1 counters equal to the number of fellowship counters on Banner of Kinship.
```

Solar's summary is very basic, and does not provide much information about the card beyond its basic text.

gpt-4o-mini  9/9:
```
Banner of Kinship is a strong artifact that enhances your creatures of a chosen type
As it enters the battlefield, you select a creature type, and it enters with fellowship counters based on how many creatures of that type you control
Each of your creatures of the chosen type gets a boost of +1/+1 for each counter on the Banner
Its mana cost of 5 is considered somewhat high, but its potential to strengthen a wide array of creatures can be very impactful, especially in creature-focused decks
However, its effectiveness is dependent on having multiple creatures of the chosen type on the battlefield, making it less reliable in decks with a diverse creature lineup.
```

gpt-4o  9/9:
```
Banner of Kinship is a colorless artifact card with a moderate mana cost that focuses on tribal synergy
When it enters the battlefield, you choose a creature type, and the artifact gains a fellowship counter for each creature of that type you control
These counters then enhance the power and toughness of creatures you control of the chosen type, making them stronger
The card is excellent in decks that center around tribal themes, increasing the utility and effectiveness of your creatures as their numbers grow
However, its effectiveness heavily depends on having multiple creatures of the chosen type, making it less impactful if your board presence is minimal or diverse in types.
```

Both GPT models provide excellent summaries, and clearly understand the role of the card.

qwq:32b Ungraded:
```
Banner of Kinship is an interesting artifact from Magic: The Gathering that can significantly boost your creatures of a specific type
Let's break down what it does and how you might use it in your deck
 First off, Banner of Kinship costs five colorless mana to cast, which is represented by {5} in its mana cost
This makes it a mid-range spell in terms of cost; not too cheap that it clogs up early game development, but also not so expensive that it's unplayable in the late game
The fact that it's an artifact means it can be cast in any deck, regardless of color identity, which adds to its versatility
 When you cast Banner of Kinship, you get to choose a creature type as it enters the battlefield
This choice is crucial because the benefits of the card are tied to this type
For example, if you choose Human, then all Human creatures you control will benefit from the card
 The card enters the battlefield with a number of fellowship counters equal to the number of creatures you currently control of the chosen type
So, if you have three Humans on the battlefield when you cast Banner of Kinship and choose Human, the artifact will enter with three fellowship counters
 The key ability of Banner of Kinship is that it grants +1/+1 to each of your creatures of the chosen type for each fellowship counter on the artifact
So, using the previous example, if you have three fellowship counters on Banner of Kinship, all your Humans would get +3/+3 (since 3 counters × +1/+1 per counter = +3/+3)
This can be a significant boost and can turn mediocre creatures into real threats
 One important thing to note is that the fellowship counters are based on the number of creatures of the chosen type you control when Banner of Kinship enters the battlefield
If you gain control of more creatures of that type later, or if some die, the number of counters doesn't change
Therefore, it's beneficial to have as many creatures of the chosen type on the battlefield as possible when you cast Banner of Kinship
 This card is particularly powerful in decks that focus on a specific creature type
For example, if you're playing a Human tribal deck, Banner of Kinship can provide a substantial boost to all your Humans, making them much stronger attackers and blockers
Similarly, in a Merfolk or Elf tribal deck, it can be a game-changer
 However, there are some weaknesses to consider
Since the card is an artifact, it's susceptible to artifact removal spells, which are common in Magic
Additionally, if your opponent has ways to remove artifacts or negate their abilities, your Banner of Kinship won't be able to provide the boosts you're hoping for
 Another consideration is that choosing the right creature type is crucial
If you choose a type that isn't well represented on your battlefield, you won't get many counters, and thus, less benefit from the card
Therefore, planning ahead and ensuring you have multiple creatures of a specific type before casting Banner of Kinship is important
 In summary, Banner of Kinship is a powerful artifact that can significantly enhance your creatures of a chosen type by granting them +1/+1 for each fellowship counter on the artifact, which is determined by the number of creatures of that type you control when it enters the battlefield
It's most effective in decks that focus on a specific creature type and have multiple creatures of that type on the battlefield when Banner of Kinship is cast
However, its effectiveness can be diminished by artifact removal or if the chosen creature type isn't well represented.
```

I want to like this model, since it clearly goes through a self-reasoning phase, however the final summary is diluted throughout the reasoning, and difficult to extract without a second pass. DeepSeek R1 does a similar process, but constrains its thinking to within dedicated markers, making it easier to extract the final summary.

mistral-small:24b Ungraded:
```
 **Banner of Kinship** - Role: Enchantment-like Artifact that boosts creatures of a specific type
- Strengths: Strong synergy with creature decks; can scale up in power as more creatures enter the battlefield
- Weaknesses: Ineffective if there are no creatures of the chosen type or if removed early
- Mana Cost: Expensive, but worth it for the right deck.
```

Extremely basic, and does not provide much information about the card, and also makes a mistake in calling it an enchantment, rather than an artifact, and in stating that the number of counters increases as more creatures enter. Instead the number is fixed based on when it enters.

quen2.5:32 6/9:
```
Banner of Kinship is a colorless artifact with a moderate mana cost of {5}
This card allows you to choose a creature type when it enters the battlefield and grants a +1/+1 bonus for each fellowship counter on Banner of Kinship to all of your creatures of that chosen type
The number of counters is determined by how many creatures of the selected type you control at the time of casting
It's strong in decks with a variety of creatures of the same type, but its effectiveness relies heavily on having multiple creatures of the designated type on the battlefield beforehand.
```

qwen's summary is good, but a bit short, and skimps on describing the role of the card.

deepseek-r1:14b 9/9:
``` 
**Banner of Kinship**  - **Mana Cost:** {5} (moderate) - **Type:** Artifact - **Effect:** When played, choose a creature type and place a fellowship counter for each creature of that type you control
Creatures of the chosen type gain +1/+1 per counter on Banner
- **Strengths:** Boosts creatures of a selected type significantly; versatile and accessible as it's colorless
- **Weaknesses:** Vulnerable to artifact removal; effectiveness depends on having enough creatures of the chosen type
- **Role:** Mid-game card, useful for creature-based decks looking to enhance their strategy
 This artifact is powerful in creature-focused decks, offering conditional boosts that can significantly strengthen your forces if utilized effectively.
```

R1's summary is very good, and captures all the information about the card and its role.

### Card addition

Based on the query, we can now search for similar cards in the database. To allow for flexibility, I set the system to instead return the top 5 cards that match the query. An LLM then selects the final card to add to the deck, based on the current state of the deck, the analysis of the deck, and the requested card coming from the request subsystem.

As an example of this, the card requester asks for a "A black sorcery that returns a target creature card from your graveyard to the battlefield"

The query process finds:

1. **Zombify**  - **Mana Cost:** {3}{B} - Moderately inexpensive.    - **Type:** Sorcery  - **Functionality:** Zombify allows you to return a target creature card from your graveyard directly to the battlefield.
1. **Come Back Wrong**: This black sorcery with a moderate mana cost allows you to destroy any target creature on the battlefield. If that destruction causes a creature card to be put into a graveyard, it returns to the battlefield under your control but requires you to sacrifice it at the beginning of your next end step.
1. **Macabre Reconstruction**  This is a black sorcery that offers versatility and synergy in gameplay.
It costs {3}{B}, which is moderately expensive but can be reduced by {2} if you've already put a creature card into your graveyard this turn, making it more accessible under the right conditions.
1. **Mourner's Surprise**  A black two-mana sorcery card that offers both graveyard utility and token generation.
It allows you to return up to one target creature card from your graveyard to your hand, which can be useful for replaying powerful creatures or enabling flashback effects.
1. **Helping Hand**  Mana Cost: {W} (Weak, white)  Role: Supportive Enchantment  This sorcery allows you to return a creature card with a mana value of 3 or less from your graveyard directly onto the battlefield tapped.
It's particularly useful for reviving small creatures that can offer immediate assistance in combat or other abilities without waiting until your next turn.

The LLM then selects Zombify, and provides the following reasoning:

```
Selected card index: 0, reason: The best card to add is Zombify (index 0). It fits well with the graveyard reanimation theme and provides a flexible way to bring creatures back from the graveyard, enhancing both board presence and deck utility. Additionally, it complements existing cards like Emerge from the Cocoon by offering another layer of creature recursion, which can be strategically powerful in long games or against specific opponents. Zombify also addresses some weaknesses by providing an alternative method for reanimation, reducing reliance on other spells that might have their own vulnerabilities. Its moderate mana cost aligns well with the current manabase and doesn't introduce new color requirements beyond what's already present. Overall, adding Zombify would provide a strong, versatile tool that supports the deck's primary strategy while fitting seamlessly into its existing card pool.
```

We can now add this card to our deck, reanalyse, and repeat the process.

## Deck mana-base fixing

The card-addition step works, however in testing, I found that the model would often forget to add lands to the deck and would end up with a deck that was unplayable. To fix this, and also to save some time, I added an initial process that aims to build the mana-base of the deck.

This involves a single step, in which an LLM takes the deck prompt and adds about 18 lands of the appropriate colours to the deck. This is based on the assumption of colour archetypes and the kinds of cards that the card selector subsystem will try to add to the deck, and what cards actually exist in the database.

Generally, for a 60-card deck, about 24 lands will be necessary, but the card selector subsystem will also add lands to the deck, so we can start with a smaller number of lands.

Following this initial mana-base construction, the card selector subsystem can then add cards to the deck. I stop the process early and reanalyse the deck, to ensure that the mana-base is correct, and that the deck is playable. This second mana-base fixing step is necessary, since the card selector subsystem may add cards that require different colours of mana, and the initial mana-base construction may not be correct. This process also allows a varying number of lands to enter the deck, which can be useful for certain strategies, e.g. aggressive decks may want fewer lands, and control decks may want more.

Importantly, while the initial mana-base construction step uses generic lands that will always exist, the second mana-base fixing step uses a search query process, in order to get set-specific special lands that may be necessary for the deck. This means that the deck can be more flexible, and can include lands that are specific to the current standard format.

After this second mana-base fixing step, the card selector subsystem can then add cards to the deck, in order to fill any remaining slots.

## Deck evaluation

Let's look at an example of a deck that was generated using the system:

I requested "A graveyard-based reanimator deck that focusses on putting large creatures into play from the graveyard", and received:
```
The Unyielding Tide

## lands: 22
- 7 x red land: red land
- 7 x black land: black land
- 3 x red_black land: red_black land
- 4 x Bloodfell Caves: 
- 1 x Jungle Hollow: 

## sorceries: 22
- 4 x Emerge from the Cocoon
- 2 x Cemetery Recruitment
- 4 x Zombify
- 3 x Come Back Wrong
- 1 x Defossilize
- 1 x Mourner's Surprise
- 3 x Squirming Emergence
- 3 x Badlands Revival
- 1 x Vat Emergence

## creatures: 15
- 3 x Cult Conscript
- 1 x Tinybones, Bauble Burglar
- 1 x Hidetsugu and Kairi
- 1 x Hoverstone Pilgrim
- 1 x Chupacabra Echo
- 1 x Darkstar Augur
- 2 x Reconstructed Thopter
- 1 x Red Herring
- 1 x Malevolent Chandelier
- 1 x Spinewoods Paladin
- 1 x Quilled Greatwurm
- 1 x Reclamation Sage

## artifacts: 1
- 1 x Buried Treasure

## Mana production: green red black
## Mana requirements: black red blue green white

### Deck Analysis:

#### **Current Strengths:**
1. **Graveyard Synergy:** The deck has a strong focus on reanimation and graveyard interactions, which is a powerful theme for creating aggressive and resilient strategies.
2. **Flexible Mana Production:** With Bloodfell Caves, Jungle Hollow, and Badlands Revival providing both black and red mana, the deck can adapt to different spell requirements, enhancing its flexibility in gameplay.
3. **Card Utility:** Cards like Defossilize, Squirming Emergence, and Vat Emergence offer versatile utility for reanimation, recursion, and board manipulation, making them valuable tools for controlling the game state.
4. **Token Generation:** Tinybones, Bauble Burglar and Mourner's Surprise provide opportunities to generate tokens that can bolster the board presence or disrupt opponents' strategies.
5. **Aggressive Creatures:** Cards like Cult Conscript, Darkstar Augur, and Red Herring offer aggressive creature options that can apply pressure early in the game.

#### **Current Weaknesses:**
1. **Lands Limitation:** The reliance on dual lands (Bloodfell Caves, Jungle Hollow) may limit mana flexibility if both colors are needed simultaneously.
2. **Graveyard Dependency:** Many key cards require creatures or permanents to be in the graveyard, which can be a double-edged sword; while it provides utility, it also requires careful management of the board state.
3. **High-Mana Cost Cards:** Some critical cards like Quilled Greatwurm and Hidetsugu and Kairi have high mana costs, potentially slowing down early game development.
4. **creature Weaknesses:** Some creatures have low toughness or vulnerability to removal, making them susceptible to opponent strategies focused on removing threats.
5. **Limited Synergy:** While the deck has strong individual cards, there is limited synergy between them, which could hinder overall deck performance in competitive play.

#### **Strategic Recommendations:**
1. **Graveyard Optimization:** Ensure that key creatures and spells are positioned to maximize the use of graveyard resources. Consider adding cards like Golgari Charm or Gaea's Harvest for additional mana fixation.
2. **Mana Curve Adjustment:** Balance high-mana cost cards with cheaper options to maintain a smooth curve, ensuring the deck remains competitive in the early game.
3. **creature Support:** Enhance the survivability of creatures through abilities that boost toughness or provide evasion (e.g., Darkstar Augur's offspring ability).
4. **Board Control:** Add more board control effects like Reclamation Sage or Malevolent Chandelier to disrupt opponents' strategies and maintain a favorable board state.
5. **Card Draw Enhancement:** Consider adding cards like Demonic Tutor for additional card draw, which can complement existing card draw mechanics in the deck.

By addressing these strengths and weaknesses and implementing strategic adjustments, the deck can become more competitive and versatile in various matchups.
```

The deck is playable, and sticks to the theme of the prompt, possibly to a fault. It also has a high number of colours in its manabase (normally up to 3 colours is recommended), and severely lacks instant-speed interaction. Unfortunately, it also fails in cashing in on the main benefit of graveyard-based strategies, which is the ability to cheat on mana costs and put large creatures into play early. I would have added creatures like Atraxa, Valgovoth, or Sire of the Seven Deaths. Other missing pieces are ways to quickly fill the player's graveyard, through self-milling and discard. 

## Conclusion

The current system runs, but has limited potential. The key bottleneck is the card selector relying on its pre-learnt knowledge of Magic, coming from its training data. This means that the system is unable to adapt to new strategies, or to new cards that are released. The system also has a limited understanding of the game, and is unable to think through complex interactions between cards. This means that the system is unable to build decks that are truly competitive, and that can adapt to the metagame.

Potential ways around this are to pre-analyse the available cards, in order to highlight to the selector what new mechanics are available, and possible cards of interest. Another way would be to include example decks from competitions and current metagame, such that the system can learn from these and build decks that are more competitive, or focus on beating existing strategies.

The search system is also a bit basic. It would be better to use meta-data filtering to e.g. only allow it to match against cards of the requested type or colour.

I think that at the end, also, there should be a process of card removal and swapping, in order to refine the deck to reduce, e.g. the number of colours, and address glaring weaknesses.

The speed of generation, it also quite slow: using R1:8b for the deck building, takes ~10m to make a 60-cards deck on a local 4090 RTX.

Two problems I found in using small, local models both stem from their capability to reason: I initially tried to build a system based around tool calling, in which an agent could decide to search for cards, add and remove cards, and evaluate the deck against others. However, the models were not able to properly call the tools, and would give up very early. This is a limitation of the models, and I think that a more powerful model would be able to reason through the process of deck-building, and be able to make more complex decisions.

The second problem is the necessary use of structured responses, in which the models return TypedDicts or Pydantic Datamodels. This ensures that the model responses contain the expected information, e.g. for index selection, or manabase definition. Sometimes models fail to return the expected information and I had to account for this, by either rerunning the model, or falling back to a simpler request.

GPT-4o, via the OpenAI API, seems to handle this very well, though, and I think that if I were to continue this project, I would move away from running local models, and instead use the API. That said, I wrote most of this before Deepseek R1 was released, and that could be a good alternative to GPT-4o, since it is much cheaper to run. However, R1 does not support tool calling or structured responses, so instead I am relying on a modified version of R1 that does support these features (MFDoom/deepseek-r1).

All in all, this project was very much a first attempt, and also meant to double as a researching and learning process for myself. The modularity coming from different function calls also means that I'm free to add in new tools, or to change the way the system works, without having to rewrite the entire system. So I expect I'll continue to improve on this, and to add in new features as I go along.

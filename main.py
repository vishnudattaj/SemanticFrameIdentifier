import nltk.tree.tree
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from nltk.corpus import wordnet as wn
import re

"""
Frames used:
1. Capital_stock
2. Commercial_transaction
3. Businesses
"""

# Function fixes NER list by adding any misidentified entities
def returnEntities(t):
    # Initializes the entity list, and a separate list for money (I have to manually create money entities)
    entityList = []
    moneyTracker = []
    # Sets current index of the iteration to 0
    index = 0
    # Sets index of last element we called to -1 (this will be useful for our money entities)
    lastChildIndex = -1
    # Iterate through the entity list
    for child in t:
        # At the start of each loop iteration
        if index - lastChildIndex > 1 and moneyTracker:
            # Flush the previous money entity
            if any(token[0].lower() in ["$", "dollars", "bucks"] for token in moneyTracker if isinstance(token, tuple)):
                entityList.append(Tree("MONEY", moneyTracker))
            moneyTracker = []
            lastChildIndex = -1
        # If the item can be found within our gazetteer...
        if str(child) in gazetteer.keys():
            # For some reason numbers from 10-99 are marked as JJ if they are written in plain english...
            # If the item's part of speech is marked as JJ...
            if 'JJ' in child:
                # We switch the current definition of the child to the improved one in the gazetteer. Indices are
                # used to cut off parentheses
                child = gazetteer[str(child)][1:-1]
                # Remove any quotations and split the new definition into the word and the POS
                child = child.replace("\'", "").split(", ")
                # The child element now is a tuple, with the word on the left and the POS on the right (the same
                # format as POS tagging)
                # Note that for the remainder of this iteration in the for loop, child will refer to this updated value
                child = (child[0], child[1])
            # If the item is a tree (meaning it is an entity)
            elif type(child) == nltk.tree.tree.Tree:
                # Switch the definition with the gazetteer and remove parentheses
                child = gazetteer[str(child)][1:-1]
                # For entities we split by the space
                child = child.split(" ")
                # First index is the type of entity is it (person, organization, etc.)
                entityType = child[0]
                # All following indexes are the words used to make up the entity (see line 4 of the gazetteer)
                items = child[1:]
                # Initialize a list to contains all the POS items within the tree
                posItems = []
                # Iterate through words of entity
                for item in items:
                    # Split by the slash to separate word from POS
                    splitItem = item.split("/")
                    # Append word and POS as a tuple to the posItems list
                    posItems.append((splitItem[0], splitItem[1]))
                # Create a tree element with entity type and the words that represent it
                # Note that for the remainder of this iteration in the for loop, child will refer to this updated value
                child = Tree(entityType, posItems)
                # Add the entity to the entity list
                entityList.append(child)
            # If it's not an entity and doesn't have JJ as its POS...
            else:
                # Switch the definition with the gazetteer and remove parentheses
                child = gazetteer[str(child)][1:-1]
                # The only other types of switches in the gazetteer are objects that are entities but were not identified as such...
                # Therefore, we split by the space
                child = child.split(" ")
                # Split by the slash to seperate word from POS, child[0] is the type of entity while child[1] is the word
                childPOS = child[1].split("/")
                # Note that for the remainder of this iteration in the for loop, child will refer to this updated value
                child = Tree(child[0], [(childPOS[0], childPOS[1])])
                # Add the entity to the entity list
                entityList.append(child)
        # If the item in the NER list is part of the tree class (meaning that it is an entity)...
        elif type(child) == nltk.tree.tree.Tree:
            # Append to our entity list
            entityList.append(child)
        # Now for entities related to money...
        # If the item has a dollar sign...
        if '$' in child:
            # If our money tracker is empty...
            if moneyTracker == []:
                # We append the dollar sign to the tracker
                moneyTracker.append(child)
                # We save the index of the dollar sign (important later on)
                lastChildIndex = index
            # If money tracker is not empty (indicating that another money entity is already in there)
            else:
                # We add that entity to the entity list...
                entityList.append(Tree("MONEY", moneyTracker))
                # And clear the money tracker
                moneyTracker = []
                # Add the dollar sign to our money tracker
                moneyTracker.append(child)
                # We save the index of the dollar sign
                lastChildIndex = index
        # If CD is in child (note that if a number has JJ it has now been changed to CD)...
        if 'CD' in child:
            # If money tracker is empty and the last element is adjacent to the index of this element...
            # (examples: [$][40] or [fifty] [thousand])
            if moneyTracker == [] or index - lastChildIndex == 1:
                # Append the number to the money tracker
                moneyTracker.append(child)
                # Update the last index with the current objects index
                lastChildIndex = index
        # If there is a conjunction (such as three hundred and fifty thousand)...
        elif "('and', 'CC')" == str(child):
            # If it is adjacent to the last child index...
            if index - lastChildIndex == 1:
                # Add the word to the money tracker
                moneyTracker.append(child)
                # Update the last child index
                lastChildIndex = index
        # If the word dollar is in the string...
        elif "dollar" in str(child):
            # If the indexes are next to each other...
            if index - lastChildIndex == 1:
                # Add the child to money tracker
                moneyTracker.append(child)
                # Turn money tracker into a money element
                entityList.append(Tree("MONEY", moneyTracker))
                # Makes money tracker empty
                moneyTracker = []
                # Resets last child index
                lastChildIndex = -1
        # Increments index by one
        index += 1
    # If money tracker is not empty at end of loop...
    if moneyTracker:
        # If $ or dollar in one of the tokens...
        if any(token[0].lower() in ["$", "dollars"] for token in moneyTracker if isinstance(token, tuple)):
            # We take the tracker and append it as an entity
            entityList.append(Tree("MONEY", moneyTracker))
    # Return the entity list
    return entityList

# Function used to identify whether stock frame exists in a sentence
def stockIdentifier(partSpeech):
    # Stock LU contains a list of the frame's LUs, each wordnet definition was manually found
    stockLU = ['share.n.02', 'stock.n.01', 'common.s.03', 'ordinary.a.01', 'favored.s.01']
    # For each string, turn it into an actual wordnet synset
    stockLU = [wn.synset(unit) for unit in stockLU]
    # For each word and POS in the POS tagged sentence...
    for (word, pos) in partSpeech:
        # Get the POS of the word in terms of wordnet (NLTK and Wordnet have different syntaxes)
        synsetPOS = get_synset_pos(pos)
        # Get all potential synsets for a word
        synsets = wn.synsets(word, pos=synsetPOS)
        # If we get a valid wordnet POS...
        if synsetPOS:
            # For each unit in the stockLU...
            for unitSynset in stockLU:
                # Iterate through each of the synsets for the word
                for synset in synsets:
                    # We check to make sure both the LU and word have the same POS
                    # This is because leacock-chodorow requires POS to be the same (Wu Palmer doesn't however)
                    if synset.pos() == unitSynset.pos():
                        try:
                            # Get both Wu Palmer and leacock-chodrow scores
                            wu_palmer_score = unitSynset.wup_similarity(synset)
                            leacock_chodorow_score = unitSynset.lch_similarity(synset)
                            # Only return true and the word if the scores are high
                            if wu_palmer_score > 0.85 or leacock_chodorow_score > 2.5:
                                return True, word
                        # Skip iteration if we get any errors
                        except:
                            continue
    # If no similarities are matched, return false and None as the word
    return False, None

# Function used to identify if a transaction occurs in a sentence
def transactionIdentifier(partSpeech):
    # Transaction LU contains a list of the frame's LUs, each wordnet definition was manually found
    transactionLU = ['transaction.n.01', 'buy.v.01', 'sell.v.01']
    # For each string, turn it into an actual wordnet synset
    transactionLU = [wn.synset(unit) for unit in transactionLU]
    # For each word and POS in the POS tagged sentence...
    for (word, pos) in partSpeech:
        # Get the POS of the word in terms of wordnet (NLTK and Wordnet have different syntaxes)
        synsetPOS = get_synset_pos(pos)
        # Get all potential synsets for a word
        synsets = wn.synsets(word, pos=synsetPOS)
        # If we get a valid wordnet POS...
        if synsetPOS:
            # For each unit in the transactionLU...
            for unitSynset in transactionLU:
                # Iterate through each of the synsets for the word
                for synset in synsets:
                    # We check to make sure both the LU and word have the same POS
                    # This is because leacock-chodorow requires POS to be the same (Wu Palmer doesn't however)
                    if synset.pos() == unitSynset.pos():
                        try:
                            # Get both Wu Palmer and leacock-chodrow scores
                            wu_palmer_score = unitSynset.wup_similarity(synset)
                            leacock_chodorow_score = unitSynset.lch_similarity(synset)
                            # Only return true and the word if the scores are high
                            if wu_palmer_score > 0.85 and leacock_chodorow_score > 2.5:
                                return True, word
                        # Skip iteration if we get any errors
                        except:
                            continue
    # If no similarities are matched, return false and None as the word
    return False, None

def businessIdentifier(partSpeech):
    # Business LU contains a list of the frame's LUs, each wordnet definition was manually found
    businessLU = ['depository_financial_institution.n.01', 'business.n.01', 'company.n.01', 'corporation.n.01', 'establishment.n.05']
    # For each string, turn it into an actual wordnet synset
    businessLU = [wn.synset(unit) for unit in businessLU]
    # For each word and POS in the POS tagged sentence...
    for (word, pos) in partSpeech:
        # Get the POS of the word in terms of wordnet (NLTK and Wordnet have different syntaxes)
        synsetPOS = get_synset_pos(pos)
        # Get all potential synsets for a word
        synsets = wn.synsets(word, pos=synsetPOS)
        # If we get a valid wordnet POS...
        if synsetPOS:
            # For each unit in the transactionLU...
            for unitSynset in businessLU:
                # Iterate through each of the synsets for the word
                for synset in synsets:
                    # We check to make sure both the LU and word have the same POS
                    # This is because leacock-chodorow requires POS to be the same (Wu Palmer doesn't however)
                    if synset.pos() == unitSynset.pos():
                        try:
                            # Get both Wu Palmer and leacock-chodrow scores
                            wu_palmer_score = unitSynset.wup_similarity(synset)
                            leacock_chodorow_score = unitSynset.lch_similarity(synset)
                            # Only return true and the word if the scores are high
                            if wu_palmer_score > 0.85 and leacock_chodorow_score > 2.5:
                                return True, word
                        # Skip iteration if we get any errors
                        except:
                            continue
    # If no similarities are matched, return false and None as the word
    return False, None

# This function is used to extract the stock frame elements from the sentence
def stockElementMatcher(sentence, indicator, entities, sentWords, wordPOS):
    # Track index, set to zero
    index = 0
    # People list includes all prepositions
    people = remove_duplicates([word for word, tag in wordPOS if tag == 'PRP'])
    # Add entities labeled as a person to people list
    people.extend(remove_duplicates([entity for entity in entities if entity.label() == "PERSON"]))
    # Create organizations list with entities labeled as organizations
    organizations = remove_duplicates([entity for entity in entities if entity.label() == "ORGANIZATION"])
    # These words indicate whether a stock occurs
    stockWords = ['share.n.02', 'stock.n.01']
    # Initialize the amount (as in number of shares of a stock) as a blank list
    amount = []
    # Set all the FE as None initially (this way, if an element doesn't exist, it will print None)
    shareholder = None
    stockWord = None
    stockType = None
    issuer = None
    # Iterate through the word and POS in the POS tagged sentence
    for word, tag in wordPOS:
        # If the tag is POS (meaning it indicates possession through use of an apostrophe)...
        if tag == 'POS' and index > 0:
            # If a shareholder doesn't exist...
            if not shareholder:
                # Shareholder is the word before the apostrophe
                shareholder = sentWords[index - 1]
        # The PRP$ tag indicates possessive pronouns
        elif tag == 'PRP$':
            # If shareholder doesn't exist...
            if not shareholder:
                # The shareholder is the possessive pronoun
                shareholder = word
        # If the number of items in the people list is one...
        elif len(people) == 1:
            # And if a shareholder doesn't already exist...
            if not shareholder:
                # The shareholder is the person in the people list
                shareholder = people[0]
                # If the shareholder is a NLTK entity...
                if type(shareholder) == nltk.tree.tree.Tree:
                    # We flatten it and extract just the name
                    shareholder = shareholder.leaves()
                    shareholder = [name for name, pos in shareholder]
                    shareholder = " ".join(shareholder).strip()
        # If the tag is a number...
        if tag == 'CD':
            # We append both the word and index it is in to amount
            amount.append((word, index))
        # For each word in stockWords...
        for potentialStock in stockWords:
            # Get the wordnet position of the word
            synsetPOS = get_synset_pos(tag)
            # If the POS exists...
            if synsetPOS:
                # Get all synsets for the word, and get the synset of the stock word
                synsets = wn.synsets(word, pos=synsetPOS)
                unitSynset = wn.synset(potentialStock)
                # For each synset of the word...
                for synset in synsets:
                    # If the POS matches (important for leacock-chodrow)
                    if synset.pos() == unitSynset.pos():
                        try:
                            # Get the relevancy scores of the two words
                            wu_palmer_score = unitSynset.wup_similarity(synset)
                            leacock_chodorow_score = unitSynset.lch_similarity(synset)
                            # If scores are high...
                            if wu_palmer_score > 0.85 or leacock_chodorow_score > 2.5:
                                # The word you have is the stock word
                                stockWord = word
                                # We want to exit out all loops once we have the stock word
                                break
                        except:
                            continue
            # If we already found the stock word, we exist out of the loop
            if stockWord:
                break
        # Increment the index by one
        index += 1
    # Initialize the real amount list (here we try and extract just the amounts)
    realAmount = []
    item = []
    # If amount has more than one item...
    if len(amount) > 1:
        # Initially set previous index to a random num (negative)
        previousIndex = -1
        # For each number and index in the amount...
        for num, index in amount:
            # If the current item's index is adjacent to the previous item's index...
            if index - previousIndex == 1:
                # Add the current item to the item list
                item.append(num)
            # If they are not adjacent...
            else:
                # If item already has stuff in it...
                if len(item) > 0:
                    # Append that to the realAmount list...
                    realAmount.append(" ".join(item))
                    # And make item a blank list again
                    item = []
                # Add the current num to item (regardless of if it had stuff in it or not)
                item.append(num)
            # Change previous index to the current index
            previousIndex = index
        # If item has stuff in it after all iterations are done...
        if len(item) > 0:
            # Add it to realAmount
            realAmount.append(" ".join(item))
    # If there only is one item in amount...
    elif len(amount) == 1:
        # Add it to real Amount
        realAmount = [" ".join([word for word, _ in amount])]
    # If we do have a stock word...
    if stockWord:
        # Make a list to store all index values
        indexList = []
        # Stock index is the index where the stock word is
        stockIndex = sentWords.index(stockWord)
        # If real amount is not empty...
        if realAmount:
            # For each number in real amount...
            for number in realAmount:
                # Find the index of the number
                numIndex = sentWords.index(number.split(" ")[0])
                # Append the difference between the stock Index and number index to the index list
                indexList.append(abs(stockIndex - numIndex))
            # Set real amount to the number that is closest to the stock word
            realAmount = realAmount[indexList.index(min(indexList))]
    # If real amount is a blank list change it to the value None
    if not realAmount:
        realAmount = None
    # Lower the sentence once so we don't have to multiple times
    sentenceLower = sentence.lower()
    # Check if the stock type is preferred or common (often times its none)
    if "preferred stock" in sentenceLower or "preferred share" in sentenceLower:
        stockType = "preferred"
    elif "common stock" in sentenceLower or "common share" in sentenceLower:
        stockType = "common"
    # If there is only one organization entity...
    if len(organizations) == 1:
        # Flatten the tree and make set the issuer of the stock to the organization
        issuer = organizations[0].leaves()
        issuer = [name for name, pos in issuer]
        issuer = " ".join(issuer).strip()
    # Print out the frame name, lexical unit triggered, and all frame elements
    print(f"Frame Name: {"Capital Stock"}")
    print(f"Lexical Unit: {indicator}")
    print(f"Shareholder: {shareholder}")
    print(f"Amount: {realAmount}")
    print(f"Stock: {stockWord}")
    print(f"Type: {stockType}")
    print(f"Issuer: {issuer}")
    # Print out the new line to clearly separate frames
    print()

# This function is used to extract the transaction elements from the sentence
def transactionElementMatcher(sentence, indicator, entities, sentWords, wordPOS):
    # Set money entities into their own list
    money = remove_duplicates([entity for entity in entities if entity.label() == "MONEY"])
    # Set items (as in organizations, prepositions, and people) into their own list
    items = remove_duplicates([word for word, tag in wordPOS if tag == 'PRP'])
    items.extend(remove_duplicates([entity for entity in entities if entity.label() == "PERSON"]))
    items.extend(remove_duplicates([entity for entity in entities if entity.label() == "ORGANIZATION"]))
    # Set the Frame Elements as None so it prints None if they're not set
    buy = None
    sell = None
    buyer = None
    seller = None
    good = set()
    # Set words that indicate a purchase as a list
    purchaseIndicators = ['buy.v.01', 'sell.v.01']
    # For each word and POS in the pos tagged sent...
    for (word, pos) in wordPOS:
        # Convert the NLTK POS into a wordnet POS
        synsetPOS = get_synset_pos(pos)
        # If we successfully get a wordnet POS...
        if synsetPOS:
            # Iterate through our purchase indicators
            for unit in purchaseIndicators:
                # Turn the indicator into a wordnet synset
                unitSynset = wn.synset(unit)
                # Get all synsets for the current word in the sentence
                synsets = wn.synsets(word, pos=synsetPOS)
                # For each synset in the word...
                for synset in synsets:
                    # If it has same POS as the purchase word...
                    if synset.pos() == unitSynset.pos():
                        try:
                            # Get both wu palmer and leacock chodorow scores
                            wu_palmer_score = unitSynset.wup_similarity(synset)
                            leacock_chodorow_score = unitSynset.lch_similarity(synset)
                            # If scores are high...
                            if wu_palmer_score > 0.85 and leacock_chodorow_score > 2.5:
                                # If word we use to get the positive match is related to buying...
                                if "buy" in unit:
                                    # Set buy variable as that word
                                    buy = word
                                else:
                                    # Else set sell variable as that word
                                    sell = word
                        except:
                            continue
        # If the word has stock in it add to the good list
        if "stock" in word:
            good.add("stock")
        # Similar thing if it has share
        elif "share" in word:
            good.add("share")
        # Else match and see if it relates to the word commodity
        else:
            unitSynset = wn.synset("commodity.n.01")
            synsets = wn.synsets(word, pos=synsetPOS)
            # Iterate through synsets of the word once again
            for synset in synsets:
                # If pos is the same...
                if synset.pos() == unitSynset.pos():
                    try:
                        wu_palmer_score = unitSynset.wup_similarity(synset)
                        leacock_chodorow_score = unitSynset.lch_similarity(synset)
                        # If scores are high...
                        if wu_palmer_score > 0.85 and leacock_chodorow_score > 2.5:
                            # Add the word into the good list
                            good.add(word)
                    except:
                        continue
    # We want to keep track of all indexes a person is mentioned in
    indexes = []
    # Iterate through items...
    for item in items:
        # If the item is a string
        if type(item) == str:
            # Append both the item, and a list of all indexes the item is mentioned in to indexes
            indexes.append((item, [index.start() for index in re.finditer(re.escape(item), sentence)]))
        # If it's a tree, flatten the tree first and do the same thing
        elif type(item) == nltk.tree.tree.Tree:
            person = item.leaves()
            person = [name for name, pos in person]
            person = " ".join(person).strip()
            indexes.append((person, [index.start() for index in re.finditer(re.escape(person), sentence)]))
    # If we have a buy word...
    if buy:
        try:
            # The buy index is the index of the buy word in the sentence
            buyIndex = sentWords.index(buy)
            # Buy proximity sorts the indexes list by the index closest to the buy index
            # This means that the first item in the list is the person/organization closest to the buy index
            buyProximity = sorted(indexes, key=lambda x: min(abs(ind - buyIndex) for ind in x[1]))
            buyer = buyProximity[0][0]
        except:
            buyer = None
    # The sell index logic works the exact same as the buy
    if sell:
        try:
            sellIndex = sentWords.index(sell)
            sellProximity = sorted(indexes, key=lambda x: min(abs(ind - sellIndex) for ind in x[1]))
            seller = sellProximity[0][0]
        except:
            seller = None
    # If we have a buy or sell word...
    if buy or sell:
        # If our items list has two entities in it...
        if len(items) == 2:
            # If we have a buyer and not a seller...
            if buyer and not seller:
                # The potential seller is the second person in the buy proximity (because its the only other entity present...)
                potentialSeller = buyProximity[1][0]
                # We get the index of this entity
                potentialIndex = sentWords.index(potentialSeller.split(" ")[0])
                # If the word that precedes is it is from or to
                if sentWords[potentialIndex - 1] == "from" or sentWords[potentialIndex - 1] == "to":
                    # We make it the potential seller
                    # Examples include: John bought the stock from Doug
                    # Here there is no sell word to indicate Doug is selling, but the word "from" implies that Doug is the seller
                    seller = potentialSeller
            # We use similar logic to potentially determine a buyer if there is only a seller
            if seller and not buyer:
                potentialBuyer = sellProximity[1][0]
                potentialIndex = sentWords.index(potentialBuyer.split(" ")[0])
                if sentWords[potentialIndex - 1] == "from" or sentWords[potentialIndex - 1] == "to":
                    buyer = potentialBuyer
    # Add the flattened money entities to the realMoney list (we only need the value, not the tree object)
    realMoney = []
    for money in money:
        moneyWords = [word for word, pos in money.leaves()]
        realMoney.append(" ".join(moneyWords))
    # Since this book only contains US currency, we can assume the unit is the dollar
    if realMoney:
        unit = "dollar"
    else:
        # No money means no units involved
        realMoney = None
        unit = None
    # If good is empty, set it as None
    if not good:
        good = None
    else:
        # Else change from a list to a set
        good = list(good)

    # Print out the frame name, lexical unit triggered, and all frame elements
    print(f"Frame Name: {"Commercial Transaction"}")
    print(f"Lexical Unit: {indicator}")
    print(f"Buyer: {buyer}")
    print(f"Seller: {seller}")
    print(f"Money: {realMoney}")
    print(f"Goods: {good}")
    print(f"Unit: {unit}")
    # Print out the new line to clearly separate frames
    print()

# This function is used to extract the business frame elements from the sentence
def businessElementMatcher(indicator, entities, sentWords, wordPOS):
    # Make lists for entities labeled as organizations or geopolitical entities
    businesses = remove_duplicates([entity for entity in entities if entity.label() == "ORGANIZATION"])
    locations = remove_duplicates([entity for entity in entities if entity.label() == "GPE"])
    descriptor = []
    # Append flattened business lists to realBusiness list
    realBusiness = []
    for business in businesses:
        theBusiness = business.leaves()
        theBusiness = [name for name, pos in theBusiness]
        theBusiness = " ".join(theBusiness).strip()
        # For the company, get the first word in its name
        startingName = theBusiness.split(" ")[0]
        # Get the index for that name...
        nameIndex = sentWords.index(startingName)
        # And get the words and POS for all words within 2 indexes of the company name
        clipped = wordPOS[nameIndex-2:nameIndex+2]
        # Iterate through the words near the company name
        for word, pos in clipped:
            # If the word is an adjective, append to descriptor
            if pos.startswith("J"):
                # Descriptor now contains adjectives likely used to describe the company
                descriptor.append(word)
        # Append the flattened business names to the realBusiness list
        realBusiness.append(theBusiness)
    # Add flattened locations to the locationNames list
    locationNames = []
    for loc in locations:
        locWords = [word for word, pos in loc.leaves()]
        locationNames.append(" ".join(locWords))
    # If there are no organizations, make it None
    if not realBusiness:
        realBusiness = None
    # If there are no descriptors, make it None
    if not descriptor:
        descriptor = None
    else:
        # If there are descriptors, remove any duplicates from the list
        descriptor = remove_duplicates(descriptor)
    # If no locationNames exist, set to None
    if not locationNames:
        locationNames = None

    # Print out the frame name, lexical unit triggered, and all frame elements
    print(f"Frame Name: {"Business"}")
    print(f"Lexical Unit: {indicator}")
    print(f"Business: {realBusiness}")
    print(f"Descriptor: {descriptor}")
    print(f"Place: {locationNames}")
    # Print out the new line to clearly separate frames
    print()

# Removes duplicates in lists
def remove_duplicates(list):
    # Initializes empty list
    newList = []
    # For item in original list...
    for item in list:
        # If it isn't in new list...
        if item not in newList:
            # Add it
            newList.append(item)
    # Return list without duplicates
    return newList

# Takes NLTK POS and turns into word net's POS (if it starts with a N its a noun, V for verb, etc.)
def get_synset_pos(pos):
    if pos.startswith("N"):
        return wn.NOUN
    elif pos.startswith("V"):
        return wn.VERB
    elif pos.startswith("J"):
        return wn.ADJ
    elif pos.startswith("R"):
        return wn.ADV
    return None

# Function accepts the sentence, named entities, and sentence number to print out the frames the sentence contains
def runIdentifiers(sent, namedEntities, sentNumber, wordTokenize, posTagged):
    # Status variables indicate whether frame is present in a sentence, word variables indicate the word that triggered the frame
    stockStatus, stockWord = stockIdentifier(posTagged)
    transactionStatus, transactionWord = transactionIdentifier(posTagged)
    businessStatus, businessWord = businessIdentifier(posTagged)

    # If sentence contains one of the three frames, print the sentence number and the sentence itself
    if stockStatus or transactionStatus or businessStatus:
        print(f"######################## Sentence: {str(sentNumber).zfill(4)} ########################")
        print(sent.replace("\n", " "))

        # If sentence relates to stock...
        if stockStatus:
            # Function accepts sentence, word indicator, and the entity list to print out lexical units and frame elements
            stockElementMatcher(sent, stockWord, namedEntities, wordTokenize, posTagged)

        # If sentence relates to transactions...
        if transactionStatus:
            # Function accepts sentence, word indicator, and the entity list to print out lexical units and frame elements
            transactionElementMatcher(sent, transactionWord, namedEntities, wordTokenize, posTagged)

        # If sentence relates to business...
        if businessStatus:
            # Function accepts sentence, word indicator, and the entity list to print out lexical units and frame elements
            businessElementMatcher(businessWord, namedEntities, wordTokenize, posTagged)

# Initialize an empty dictionary to store the gazetteer
gazetteer = {}

# Open the data file and use it to populate the dictionary
with open('corpusGazetteer.txt', 'r') as gazFile:
    for line in gazFile:
        # Each line in the file is an entry in the gazetteer
        entry = line.strip().split('->')
        # Enter the key and value into the gazetteer dictionary
        gazetteer[entry[0]] = entry[1]

# Open the book file, read it, and sentence tokenize it
with open('corpus.txt', 'r', encoding="utf-8") as corpus:
    corpusText = corpus.read()
    corpusSentences = sent_tokenize(corpusText)

    # Initialize variable to count sentences
    sentNumber = 1
    # Iterate through the sentences
    for sent in corpusSentences:
        # The book contains many underscores and a few asterisks to represent italics and bolded words, we remove these
        sent = sent.replace("_", "")
        sent = sent.replace("**", "")
        # Word tokenize the sentence
        corpusWord = word_tokenize(sent)
        # POS tag the words
        sentPOS = pos_tag(corpusWord)
        # Find named entities using the POS
        sentNER = ne_chunk(sentPOS)
        # Run function to add any misidentified entities to existing NER list
        namedEntities = returnEntities(sentNER)
        # Function runs identifier functions used to identify the semantic frame
        runIdentifiers(sent, namedEntities, sentNumber, corpusWord, sentPOS)
        # Increment the sentence counter by one
        sentNumber += 1

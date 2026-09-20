The following are demos of asking the model questions. Note that model will always lock onto the same word if given the exact same verbatim query.

```
.venv/bin/python ask.py "what is a car made of? "           
==============================================================================
YOU:         'what is a car made of? '
locked on:   'car' (at 'is a car', confidence 0.95)
graph found: ['car UsedFor driving', 'car IsA vehicle', 'bicycle RelatedTo car']
GRAPH OFF:   what is a car made of? 'My sent at more at more at more at more at more '
GRAPH ON:    what is a car made of? 'bentethirea. Ther shat benenenligeeeeGde. Sisted'

.venv/bin/python ask.py "cancer "                
==============================================================================
YOU:         'cancer '
locked on:   'car' (at 'cancer', confidence 0.04)
graph found: ['car UsedFor driving', 'car IsA vehicle', 'bicycle RelatedTo car']
GRAPH OFF:   cancer 'at more at more at more at more at more at more '
GRAPH ON:    cancer 'attewehardy bentehireau. I mamerver ass intetehr'

YOU:         'apple '
locked on:   'apple' (at 'apple', confidence 1.00)
graph found: ['apple HasA core', 'apple HasProperty red', 'apple MadeOf fiber']
GRAPH OFF:   apple 'assed at more at more at more at more at more at'
GRAPH ON:    apple 'ass intetehrea. Therey shat bentehirea. Therey s'

.venv/bin/python ask.py "fruit "
==============================================================================
YOU:         'fruit '
locked on:   'rabbit' (at 'fruit', confidence 0.99)
graph found: ['rabbit HasA ear', 'rabbit CapableOf hopping', 'rabbit HasProperty fluffy']
GRAPH OFF:   fruit 'must at more at more at more at more at more at '
GRAPH ON:    fruit 'sememexee�ss intes shemerveyingeeeGde. I say, th'

.venv/bin/python ask.py "flower "
==============================================================================
YOU:         'flower '
locked on:   'flower' (at 'flower', confidence 0.95)
graph found: ['flower HasA petal', 'flower CapableOf blooming', 'flower HasProperty pretty']
GRAPH OFF:   flower 'at more at more at more at more at more at more '
GRAPH ON:    flower 'seharuauy. I mamervey ass istetehreau. I mamerve'

.venv/bin/python ask.py "phone " 
==============================================================================  ## stone and phone are both ending in 'one' and I presume that is what has lead to the model locking onto stone (also phone isn't in the KB); a better trained  model may be able to have a lower confidence and hence get rejected 
YOU:         'phone '
locked on:   'stone' (at 'phone', confidence 0.78)
graph found: ['stone CapableOf eroding', 'stone HasA crystal', 'stone HasProperty hard']
GRAPH OFF:   phone 'were as more at more at more at more at more at '
GRAPH ON:    phone 'siss istethingeeeGde. I mamervey ass istethingee'

.venv/bin/python ask.py "aidjg; "
==============================================================================
YOU:         'aidjg; '
locked on:   'egg' (at 'aidjg', confidence 0.42)
graph found: ['egg HasA yolk', 'egg MadeOf protein', 'egg RelatedTo chicken']
GRAPH OFF:   aidjg; 'but the were as more at more at more at more at '
GRAPH ON:    aidjg; 'that bentey sharuain muses shearuain sistedaey, '

.venv/bin/python ask.py "cat "   
==============================================================================
YOU:         'cat '
locked on:   'cat' (at 'cat', confidence 0.96)
graph found: ['cat HasA whisker', 'cat CapableOf hunting', 'cat HasProperty agile']
GRAPH OFF:   cat 'must at more at more at more at more at more at '
GRAPH ON:    cat 'bentethirea. Therey shat benentehrerks amervey a'

.venv/bin/python ask.py "onomatopiea "
============================================================================== ## Here the model appear to lock onto both lake and table?
YOU:         'onomatopiea '
locked on:   'table' (at 'matopiea', confidence 0.08)
graph found: ['lake HasA shore', 'lake HasProperty deep', 'table UsedFor supporting']
GRAPH OFF:   onomatopiea 'the were as more at more at more at more at more'
GRAPH ON:    onomatopiea 'mareauy. I mamervey amervey amervey amervey amer'

.venv/bin/python ask.py "flowering boats "
==============================================================================
YOU:         'flowering boats '
locked on:   'clay' (at 'ng boats', confidence 0.20)
graph found: ['clay HasA particle', 'clay RelatedTo pot', 'clay UsedFor molding']
GRAPH OFF:   flowering boats 'at more at more at more at more at more at more '
GRAPH ON:    flowering boats 'seharuain. Ther sehat bentethireau. I mamervey a'

.venv/bin/python ask.py "I have some flowers on venus "
============================================================================== # When confidences are low it seems to lock onto clay
YOU:         'I have some flowers on venus '
locked on:   'sun' (at 'on venus', confidence 0.16)
graph found: ['clay HasA particle', 'clay UsedFor molding', 'clay HasProperty soft']
GRAPH OFF:   I have some flowers on venus 'at more at more at more at more at more at more '
GRAPH ON:    I have some flowers on venus 'sehat bentethirea. Ther shat bentethingeeeeGde. '

.venv/bin/python ask.py "I have some flowers on the sun "
==============================================================================
YOU:         'I have some flowers on the sun '
locked on:   'sun' (at ' the sun', confidence 0.75)
graph found: ['sun HasA ray', 'sun CapableOf shining', 'sun HasProperty bright']
GRAPH OFF:   I have some flowers on the sun 'were at more at more at more at more at more at '
GRAPH ON:    I have some flowers on the sun 'sistedain ses shears istedain ses shears istedai'

.venv/bin/python ask.py "saucepan "                      
==============================================================================
YOU:         'saucepan '
locked on:   'pan' (at 'saucepan', confidence 0.82)
graph found: ['pan UsedFor cooking', 'pan HasA handle', 'pan HasProperty hot'] 
GRAPH OFF:   saucepan 'were at more at more at more at more at more at '
GRAPH ON:    saucepan 'thars istedaeauy. I mamervey ass istedaeauy. I m'

```
The graph that was used for this test is available in graph.json 

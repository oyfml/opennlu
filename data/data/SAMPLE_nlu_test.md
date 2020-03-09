<!--- Make sure to update this training data file with more training examples from https://forum.rasa.com/t/rasa-starter-pack/704 --> 

## intent:goodbye <!--- The label of the intent --> 
- Bye 			<!--- Training examples for intent 'bye'--> 
- Goodbye
- See you later
- Bye bot
- Goodbye friend
- bye
- bye for now
- catch you later
- gotta go
- See you
- goodnight
- have a nice day

## intent:greet
- Hi
- Hey
- Hi bot
- Hey bot
- Hello
- Good morning
- hi again
- hi folks
- hi Mister
- hi pal!
- hi there
- greetings
- hello everybody
- hello is anybody there
- hello robot


## intent:self_introduction
- please introduce yourself
- tell me about you
- who are you
- what is your name
- May I know your name
- may I know about you

## intent:thanks
- appreciated
- appreciate it
- thank you so much
- thanks bye
- ty

## intent:wiki-search
- search for me [singapore national anthem](info)
- search for me []
- What is this [food in singapore](info)
- What is [national day in singapore](info)
-  I want to know something about 
- tell me about [LTA singapore](info)

## intent:emergency
- help in urgent emergency
- its an emergency
- please suggest what to do in urgency
- there is an emergency please help
- urgency please do something
- if there is any emergency, what do you do
- how can you help in urgent circumstances
- how do i get call help line center
- dial to emerency call 

## intent:station-name
- how to go to [bukit batok](name)
- i want to go [clementi](name)
- how reach [serangoon](name)
- is this going [toa payoh](name)
- need to go [habourfront](name)
- i need to go to [botanic gardens](name), can you help?
- where is [braddell](name)
- can you direct me to [jurong east](name)
- how do you get to [pioneer](name)
- can you tell me how to go [simei](name)
- how far to alight at [tampines](name)
- where to go to alight at [beauty world](name)
- can go [holland village](name) from here? how
- how to reach [punggol](name)
- how to arrive at [hougang](name)
- how do i arrive [sengkang](name) please tell
- are you going [paya lebar](name)
- do you pass by [orchard](name)


## lookup:name
- stations.txt

## lookup:search
- search.txt

# Assignment 1

Ting-Hsuan Lien

## Reflections

1. The Py-Feat detector has several stages. First it finds the faces in an image, and then it uses this information to extract other forms of data (AU activations, expressed emotion, ...).

   - Based on the visualization you produced, do you agree with all its predictions?

     No.

     The 7 emotions are 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral'. So it can't fully express all emotions.

     And in "grandparents.jpg" the grandpa is obviously positive emotions, but the prediction is 'anger'.

   - What seems to confuse the system when it fails?

     If it's no complete front face, the prediction have more fails.

   - Are there any cases that would be tricky for a human observer?

     Because human emotions are very complex, some expressions are difficult to define.

     Like "tablet.jpg" the woman might be sadness, but also can be in a pensive or difficult mood.

     

2. Based on the analysis of the AU data you have performed, suppose you need to choose a subset of the AUs as inputs for a predictive algorithm.

   - Which AUs would you choose, and why?

     AU7, AU10, AU25, AU12, AU6, AU 20

     They are more different between positive and negative emotions. So if it can only use few AUs, they will be more useful.

   - What's the problem with using *too many* features?

     Too many features need more computing resource, and sometimes it even causes over-fitting.
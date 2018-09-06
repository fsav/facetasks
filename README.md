# facetasks

Various utilities to deal with face image datasets.

## facemask.py

Uses dlib to find a face, then the facial landmarks. Using the "convex hull", the outer perimeter is computed, and transformed into a mask.

I use this to penalize only the face region in face-related tasks.

Examples, including the optional debug image that can be generated:

| Input                            |  Mask (output)                  | Debug image                     |
|----------------------------------|---------------------------------|---------------------------------|
| ![](assets/facemask/input1.jpg)  |  ![](assets/facemask/mask1.jpg) | ![](assets/facemask/debug1.jpg) |
| ![](assets/facemask/input2.jpg)  |  ![](assets/facemask/mask2.jpg) | ![](assets/facemask/debug2.jpg) |

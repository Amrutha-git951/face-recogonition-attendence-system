Face recogonition attendence system:- 
Maintaining attendance is very important in all educational institutions. Every institution has its own method of taking student attendance. Face recognition has characteristics that other biometrics do not have. Facial images can be captured from a distance and any special action is not required for authentication. Due to such characteristics, the face recognition technique is applied widely, not only to security applications but also to image indexing, image 
retrievals and natural user interfaces. Faces are highly challenging and dynamic objects that are employed as biometric evidence, in identity verification. Biometrics systems have proven to be an essential security tool, in which bulk matching of enrolled people and watch lists is performed every day.  
installations - : Download Python 3.8.3 version,  
                 Installation of OpenCV using Pip, 
                 Command to check the version of OpenCv, 
                 Installing Tkinter and Importing, 
                 Installation of DateTime and Importing, 
                 Installation of pycharm.
                 
                
The implementation is done using the Haar CascadeObjectDetector in OpenCV which detects objects using the LBPH algorithm. “The cascade object detector uses the LBPH algorithm to detect people’s faces, noses, eyes, mouth, or upper body”. The LBPH algorithm examines an image within a sliding box to match dark or light region to identify a face that contains mouth, eyes and nose. The window size varies with different faces on different scales with the ratio unchanged. The cascade classifier in OpenCV will determine the regions where a face can be 
detected. According to OpenCV, the stages in the cascade classifier are designed to rule out regions that do not have a face in the initial stage (reject negative samples) to save time to analyze regions with possible potential faces in the next stages.

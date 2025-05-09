import numpy as np
import cv2

# SEDC-T
# https://github.com/ADMAntwerp/ImageCounterfactualExplanations
# https://link.springer.com/article/10.1007/s10044-021-01055-y

class SEDCTCFE:
    @staticmethod
    def sedc_t(image, predict_fn, segments, target_class, mode):
        """
        Parameters:
            image : numpy.ndarray
                The input image to be explained, expected to be a 3D array (height, width, channels).
            predict_fn : callable
                A function that takes an image as input and returns an array of probability distribution over all classes.
            segments : numpy.ndarray
                A 2D array of the same height and width as the input image, where each unique value represents a segment of the image (e.g., generated by a segmentation algorithm).
            target_class : int
                The index of the target class for which the explanation is being generated (in binary classification it's either 0 or 1).
            mode : str
                The perturbation mode to use for modifying the image.
                - 'mean': Replace segments with the mean color of the entire image.
                - 'blur': Replace segments with a Gaussian-blurred version of the image.
                - 'random': Replace segments with random noise.
                - 'inpaint': Replace segments using inpainting to fill the region.
        
        Returns:
            explanation : numpy.ndarray or None
                A 3D array of the same shape as the input image, highlighting the segments that contribute to the target class prediction. Returns None if no explanation is found.
            segments_in_explanation : list or None
                A list of segment indices that contribute to the explanation. Returns None if no explanation is found.
            perturbation : numpy.ndarray or None
                The perturbed version of the input image corresponding to the explanation. Returns None if no explanation is found.
            new_class : int or None
                The predicted class after applying the perturbation. Returns None if no explanation is found.
        """

        # Get the prediction for the original image
        result = predict_fn(image)
        
        # Predicted class for the original image
        c = np.argmax(result)
        
        # Probability score for the target class
        p = result[target_class]
        
        # List to store explanations (segments contributing to the target class)
        R = []
        
        # List to store corresponding perturbed images
        I = []
        
        # List to store corresponding new predicted classes
        C = []
        
        # List to store corresponding scores for the target class
        P = []
        
        # List of segment combinations to expand on during the search
        sets_to_expand_on = []
        
        # Array to store the difference in probability scores for the target class
        # between the current and original predictions for each segment combination
        P_sets_to_expand_on = np.array([])
        
        # Prepare the basis perturbed image based on the selected mode
        if mode == 'mean':
            perturbed_image = np.zeros(image.shape)
            perturbed_image[:,:,0] = np.mean(image[:,:,0])
            perturbed_image[:,:,1] = np.mean(image[:,:,1])
            perturbed_image[:,:,2] = np.mean(image[:,:,2])
        elif mode == 'blur':
            perturbed_image = cv2.GaussianBlur(image, (31, 31), 0)
        elif mode == 'random':
            perturbed_image = np.random.random(image.shape)
        elif mode == 'inpaint':
            perturbed_image = np.zeros(image.shape)
            for j in np.unique(segments):
                image_absolute = image
                mask = np.full([image_absolute.shape[0], image_absolute.shape[1]], 0)
                mask[segments == j] = 255
                mask = mask.astype('uint8')
                image_segment_inpainted = cv2.inpaint(image_absolute, mask, 3, cv2.INPAINT_NS)
                perturbed_image[segments == j] = image_segment_inpainted[segments == j]
        
        # Consider each separate segment and check if it changes the prediction to the target class
        # If not, add it to the list of segments to expand on later
        for j in np.unique(segments):
            test_image = image.copy()
            test_image[segments == j] = perturbed_image[segments == j]

            result = predict_fn(test_image)
            c_new = np.argmax(result)
            p_new = result[target_class]

            if c_new == target_class:
                R.append([j])
                I.append(test_image)
                C.append(c_new)
                P.append(p_new)
            else: 
                sets_to_expand_on.append([j])
                P_sets_to_expand_on = np.append(P_sets_to_expand_on, p_new - result[c])
        
        iterations = 1
        max_iterations = 10

        # Continue expanding the segments until we find a valid explanation (if any)
        while len(R) == 0:
            if iterations > max_iterations or len(sets_to_expand_on) == 0:
                return None, None, None, None
            
            # Select the combination of segments that maximizes the probability score for the target class
            # Each combination is represented by a list of segments, e.g. [1, 2, 3]
            # Each combination is stored in sets_to_expand_on
            # In the beginning the combinations are just single segments, e.g. [1], [2], [3]
            # They are expanded and stored in combo_set
            # They we check if they change the prediction to the target class
            # If they don't, they are added to sets_to_expand_on for further expansion in the next iterations
            # This is done by adding expanded combinations of two segments, e.g. [1, 2], [1, 3], [2, 3], to sets_to_expand_on
            # So in the next iteration sets_to_expand_on will contain both combinations of one segment and combinations of two segments
            combo = np.argmax(P_sets_to_expand_on)
            combo_set = []
            for j in np.unique(segments):
                if j not in sets_to_expand_on[combo]:
                    combo_set.append(np.append(sets_to_expand_on[combo], j))
            
            # Pruning step
            # Prevents infinite loop in case no better explanation is found after adding new segments
            del sets_to_expand_on[combo]
            P_sets_to_expand_on = np.delete(P_sets_to_expand_on, combo)
            
            for cs in combo_set: 
                test_image = image.copy()

                for k in cs: 
                    test_image[segments == k] = perturbed_image[segments == k]
                    
                result = predict_fn(test_image)
                c_new = np.argmax(result)
                p_new = result[target_class]
                    
                if c_new == target_class:
                    R.append(cs)
                    I.append(test_image)
                    C.append(c_new)
                    P.append(p_new)
                else: 
                    sets_to_expand_on.append(cs)
                    P_sets_to_expand_on = np.append(P_sets_to_expand_on, p_new - result[c])
            
            print(f"Iteration {iterations}")
            iterations += 1
                
        # Select best explanation, i.e. highest target score increase
        best_explanation = np.argmax(P - p) 
        segments_in_explanation = R[best_explanation]
        explanation = np.full([image.shape[0], image.shape[1], image.shape[2]], 0)
        for i in R[best_explanation]:
            explanation[segments == i] = image[segments == i]
        perturbation = I[best_explanation]
        new_class = C[best_explanation]
        
        return explanation, segments_in_explanation, perturbation, new_class 

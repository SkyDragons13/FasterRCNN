package com.example.frontend.controller;

import com.example.frontend.database.entity.ImageEntity;
import com.example.frontend.service.ImageService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

import java.io.IOException;
import java.security.Principal;

@Controller
@RequestMapping("/inference")
public class InferenceController {

    private final ImageService imageService;
    private final Logger logger = LoggerFactory.getLogger(InferenceController.class);

    @Autowired
    public InferenceController(ImageService imageService) {
        this.imageService = imageService;
    }

    @GetMapping
    public String showInferencePage(Model model, Principal principal) {
        model.addAttribute("username", principal.getName());
        return "user/inference";
    }

    @PostMapping("/upload")
    public String uploadImage(@RequestParam("file") MultipartFile file,
                              @RequestParam("name") String imageName,
                              RedirectAttributes redirectAttributes) {
        try {
            logger.info("Received image upload request: name={}, size={} bytes", imageName, file.getSize());

            ImageEntity image = new ImageEntity();
            image.setName(imageName);
            image.setImage(file.getBytes());

            imageService.save(image);

            Long imageId = image.getId();
            logger.info("Starting inference process for image ID: {}", imageId);


            ProcessBuilder pb = new ProcessBuilder("C:/Users/csere/anaconda3/python.exe", "C:/##FACULTATE/Anul 3 Semestru 1/Practica/inference.py", String.valueOf(imageId));

            Process process = pb.start();
            int exitCode = process.waitFor();

            if (exitCode == 0) {
                logger.info("Inference completed successfully for image ID: {}", imageId);
                redirectAttributes.addFlashAttribute("message", "Inference completed successfully.");
            } else {
                logger.warn("Inference failed for image ID: {} with exit code: {}", imageId, exitCode);
                redirectAttributes.addFlashAttribute("message", "Inference failed.");
            }
            return "redirect:/results/" + image.getId();

        } catch (IOException | InterruptedException e) {
            logger.error("Failed to upload image", e);
            redirectAttributes.addFlashAttribute("message", "Internal error during inference.");
            return "redirect:/inference?error=upload_failed";
        }
    }
}

package com.example.frontend.controller;

import com.example.frontend.database.entity.AnnotatedImageEntity;
import com.example.frontend.database.entity.ImageEntity;
import com.example.frontend.service.AnnotatedImageService;
import com.example.frontend.service.ImageService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.ResponseBody;

import java.util.Optional;

@Controller
public class ResultsController {

    private final ImageService imageService;
    private final AnnotatedImageService annotatedImageService;

    @Autowired
    public ResultsController(ImageService imageService, AnnotatedImageService annotatedImageService) {
        this.imageService = imageService;
        this.annotatedImageService = annotatedImageService;
    }

    @GetMapping("/results/{originalId}")
    public String showResults(@PathVariable Long originalId,
                              Model model) {
        ImageEntity original = imageService.findById(originalId);
        Optional<AnnotatedImageEntity> annotated = annotatedImageService.findAnnotatedByOriginal(originalId);

        if (original == null  ||  annotated.isEmpty() ) {
            return "redirect:/error";
        }

        model.addAttribute("originalImageId", original.getId());
        model.addAttribute("annotatedImageId", annotated.get().getId());

        return "user/results";
    }
    @GetMapping("/results/image/{id}")
    @ResponseBody
    public ResponseEntity<byte[]> getImage(@PathVariable Long id) {
        ImageEntity image = imageService.findById(id);
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.IMAGE_JPEG);
        return new ResponseEntity<>(image.getImage(), headers, HttpStatus.OK);
    }
    @GetMapping("/results/annotated_image/{id}")
    @ResponseBody
    public ResponseEntity<byte[]> getAnnotatedImage(@PathVariable Long id) {
        AnnotatedImageEntity image = annotatedImageService.findById(id).get();
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.IMAGE_JPEG);
        return new ResponseEntity<>(image.getImage(), headers, HttpStatus.OK);
    }
}

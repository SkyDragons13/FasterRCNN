package com.example.frontend.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import com.example.frontend.service.ImageService;
import com.example.frontend.database.entity.ImageEntity;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.ResponseBody;

import java.util.List;

@Controller
public class HistoryController {

    private final ImageService imageService;

    @Autowired
    public HistoryController(ImageService imageService) {
        this.imageService = imageService;
    }

    @GetMapping("/history")
    public String showHistory(Model model) {
        List<ImageEntity> images = imageService.findAll();
        model.addAttribute("images", images);
        return "user/history";
    }

    @GetMapping("/history/image/{id}")
    @ResponseBody
    public ResponseEntity<byte[]> getImage(@PathVariable Long id) {
        ImageEntity image = imageService.findById(id);
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.IMAGE_JPEG);
        return new ResponseEntity<>(image.getImage(), headers, HttpStatus.OK);
    }
}

package com.example.frontend.service;

import com.example.frontend.database.entity.ImageEntity;
import com.example.frontend.database.repository.ImageRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Base64;
import java.util.List;
@Service
public class ImageService {
    private final ImageRepository imageRepository;

    @Autowired
    public ImageService(ImageRepository imageRepository) {
        this.imageRepository=imageRepository;
    }

    public List<ImageEntity> findAll(){return this.imageRepository.findAll();}

    public ImageEntity findById(Long id){
        return this.imageRepository.findById(id).
                orElseThrow(() -> new RuntimeException("Image not  not found with ID: " + id));}

    public void delete(ImageEntity image)
    {
        this.imageRepository.delete(image);
    }
    public void save(ImageEntity image)
    {
        this.imageRepository.save(image);
    }
    public String convertImageToBase64(ImageEntity imageEntity) {
        byte[] imageBytes = imageEntity.getImage();
        return Base64.getEncoder().encodeToString(imageBytes);
    }
}
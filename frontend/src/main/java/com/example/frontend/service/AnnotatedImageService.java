package com.example.frontend.service;

import com.example.frontend.database.entity.AnnotatedImageEntity;
import com.example.frontend.database.entity.ImageEntity;
import com.example.frontend.database.repository.AnnotatedImageRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Optional;

@Service
public class AnnotatedImageService {

    private final AnnotatedImageRepository annotatedImageRepository;

    @Autowired
    public AnnotatedImageService(AnnotatedImageRepository annotatedImageRepository) {
        this.annotatedImageRepository = annotatedImageRepository;
    }

    public AnnotatedImageEntity save(AnnotatedImageEntity annotatedImage) {
        return annotatedImageRepository.save(annotatedImage);
    }

    public Optional<AnnotatedImageEntity> findById(Long id) {
        return annotatedImageRepository.findById(id);
    }

    public Optional<AnnotatedImageEntity> findAnnotatedByOriginal(Long originalImageId) {
        return annotatedImageRepository.findByOriginalImageId(originalImageId);
    }
}

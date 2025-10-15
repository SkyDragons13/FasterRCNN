package com.example.frontend.service;

import com.example.frontend.database.entity.AnnotationsEntity;
import com.example.frontend.database.repository.AnnotationsRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class AnnotationsService {

    private final AnnotationsRepository annotationsRepository;

    @Autowired
    public AnnotationsService(AnnotationsRepository annotationsRepository) {
        this.annotationsRepository = annotationsRepository;
    }

    public List<AnnotationsEntity> findAll() {
        return this.annotationsRepository.findAll();
    }

    public AnnotationsEntity findById(Long id) {
        return this.annotationsRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Annotation not found with ID: " + id));
    }

    public void delete(AnnotationsEntity annotation) {
        this.annotationsRepository.delete(annotation);
    }

    public void save(AnnotationsEntity annotation) {
        this.annotationsRepository.save(annotation);
    }
}

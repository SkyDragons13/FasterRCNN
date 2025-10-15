package com.example.frontend.database.repository;

import com.example.frontend.database.entity.AnnotatedImageEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface AnnotatedImageRepository extends JpaRepository<AnnotatedImageEntity,Long> {
    Optional<AnnotatedImageEntity> findByOriginalImageId(Long originalImageId);

}

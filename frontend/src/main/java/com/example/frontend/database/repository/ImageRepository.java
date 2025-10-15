package com.example.frontend.database.repository;

import com.example.frontend.database.entity.ImageEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
@Repository
public interface ImageRepository extends JpaRepository<ImageEntity,Long> {
}

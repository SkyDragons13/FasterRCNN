package com.example.frontend.database.repository;

import com.example.frontend.database.entity.AnnotationsEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
@Repository
public interface AnnotationsRepository extends JpaRepository<AnnotationsEntity,Long> {
}

package com.example.frontend.database.entity;

import jakarta.persistence.*;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.Setter;

@Entity
@Getter @Setter @EqualsAndHashCode
@Table(name = "annotated_images")
public class AnnotatedImageEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String name;

    @Column(nullable = false)
    private byte[] image;

    @ManyToOne(optional = false)
    @JoinColumn(name = "original_image_id", nullable = false,
            foreignKey = @ForeignKey(name = "fk_original_image"))
    private ImageEntity originalImage;

    // Constructors
    public AnnotatedImageEntity() {}

    public AnnotatedImageEntity(String name, byte[] image) {
        this.name = name;
        this.image = image;
    }


}

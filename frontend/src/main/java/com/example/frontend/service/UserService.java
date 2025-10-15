package com.example.frontend.service;

import com.example.frontend.database.entity.UserEntity;
import com.example.frontend.database.repository.UserRepository;
import jakarta.transaction.Transactional;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class UserService {

    private final UserRepository userRepository;

    @Autowired
    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;

    }
    public List<UserEntity> findAllByUsername(String username) {
        return userRepository.findAllByUsername(username);
    }

    public Optional<UserEntity>  findByEmail(String email) {
        return userRepository.findByEmail(email);
    }

    public boolean existsByEmail(String email) {
        return userRepository.existsByEmail(email);
    }

    @Transactional
    public UserEntity save(UserEntity user) {
        user.setPassword(new BCryptPasswordEncoder().encode(user.getPassword()));
        return userRepository.save(user);
    }

    @Transactional
    public void updateUser(Long id, String username, String email, String password) {
        userRepository.updateUserEntity(id, username, email, password);
    }
}

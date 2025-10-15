package com.example.frontend.controller;

import com.example.frontend.database.entity.UserEntity;
import com.example.frontend.service.UserService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
@RequestMapping("/login")
public class LoginController {
    private UserService userService;
    private Logger logger = LoggerFactory.getLogger(RegisterController.class);

    @Autowired
    public LoginController(UserService userService) {
        this.userService = userService;
    }

    @GetMapping()
    public String loadLoginPage(Model model){
        model.addAttribute("user", new UserEntity());
        return "authentication/login";
    }
}

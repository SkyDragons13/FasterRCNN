package com.example.frontend.controller;

import com.example.frontend.database.entity.UserEntity;
import com.example.frontend.service.UserService;
import com.example.frontend.service.UserValidatorService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;

import java.util.Optional;

@Controller
@RequestMapping("/register")
public class RegisterController {
    private UserService userService;
    private UserValidatorService userValidatorService;
    private Logger logger = LoggerFactory.getLogger(RegisterController.class);

    @Autowired
    public RegisterController(UserService userService, UserValidatorService userValidatorService) {
        this.userService = userService;
        this.userValidatorService = userValidatorService;
    }

    @GetMapping()
    public String loadRegisterPage(Model model){
        model.addAttribute("user", new UserEntity());
        return "authentication/register";
    }

    @PostMapping()
    public String register(@ModelAttribute("user") UserEntity user, BindingResult bindingResult){
        userValidatorService.validate(user, bindingResult);

        if (bindingResult.hasErrors())
            return "authentication/register";

        if (userService.existsByEmail(user.getEmail())) {
            bindingResult.rejectValue("email", "user.isEmailTaken");
            return "authentication/register";
        }

        user.setEmail(user.getEmail());
        user.setUsername(user.getUsername());

        userService.save(user);

        return "redirect:/login";
    }
}

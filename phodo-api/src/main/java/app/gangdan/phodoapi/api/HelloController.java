package app.gangdan.phodoapi.api;

import io.swagger.annotations.ApiOperation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.validation.BindingResult;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import springfox.documentation.annotations.ApiIgnore;

@Tag(name = "hello", description = "테스트 API")
@RequiredArgsConstructor
@RestController
@RequestMapping("/api/hello")
public class HelloController {

    @Tag(name = "members")
    @ApiOperation(value = "test!!")
    @GetMapping("")
    public ResponseEntity<?> testHello(){

        return new ResponseEntity<>(HttpStatus.OK);
    }
}

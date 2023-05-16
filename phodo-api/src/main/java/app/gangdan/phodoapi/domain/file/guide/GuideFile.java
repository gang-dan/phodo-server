package app.gangdan.phodoapi.domain.file.guide;

import app.gangdan.phodoapi.domain.file.File;
import app.gangdan.phodoapi.domain.photoGuide.PhotoGuide;
import lombok.Getter;
import lombok.ToString;

import javax.persistence.*;

@Getter
@ToString(exclude = {"photoGuide"})
@DiscriminatorValue("photoGuide")
@Entity
public class GuideFile extends File {

    @OneToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "photo_guide_id", nullable = false)
    private PhotoGuide photoGuide;

    protected GuideFile(){}

    public GuideFile(PhotoGuide photoGuide, String fileUrl){
        super(fileUrl);
        this.photoGuide = photoGuide;
    }
}

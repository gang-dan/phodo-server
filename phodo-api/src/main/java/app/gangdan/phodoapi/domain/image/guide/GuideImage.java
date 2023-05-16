package app.gangdan.phodoapi.domain.image.guide;

import app.gangdan.phodoapi.domain.image.Image;
import app.gangdan.phodoapi.domain.photoGuide.PhotoGuide;
import lombok.Getter;
import lombok.ToString;

import javax.persistence.*;

@Getter
@ToString(exclude = {"photoGuide"})
@DiscriminatorValue("photoGuide")
@Entity
public class GuideImage extends Image {

    @OneToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "photo_guide_id", nullable = false)
    private PhotoGuide photoGuide;

    protected GuideImage(){}

    public GuideImage(PhotoGuide photoGuide, String imagerUrl){
        super(imagerUrl);
        this.photoGuide = photoGuide;
    }
}

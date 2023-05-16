package app.gangdan.phodoapi.domain.image.mask;

import app.gangdan.phodoapi.domain.image.Image;
import app.gangdan.phodoapi.domain.photoGuide.PhotoGuide;
import lombok.Getter;
import lombok.ToString;

import javax.persistence.*;

@Getter
@ToString(exclude = {"photoGuide"})
@DiscriminatorValue("photoGuide")
@Entity
public class MaskImage extends Image {

    @OneToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "photo_guide_id", nullable = false)
    private PhotoGuide photoGuide;

    protected MaskImage(){}

    public MaskImage(PhotoGuide photoGuide, String imagerUrl){
        super(imagerUrl);
        this.photoGuide = photoGuide;
    }
}

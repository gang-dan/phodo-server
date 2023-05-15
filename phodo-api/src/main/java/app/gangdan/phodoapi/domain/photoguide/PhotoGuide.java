package app.gangdan.phodoapi.domain.photoguide;

import app.gangdan.phodoapi.domain.BaseEntity;
import lombok.*;

import javax.persistence.*;

@Entity
@Table(name = "photo_guide")
@Getter
@Builder
@NoArgsConstructor @AllArgsConstructor
public class PhotoGuide extends BaseEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long photoGuideId;

    private String PhotoGuideName;

    private Long photoSpotId;

}

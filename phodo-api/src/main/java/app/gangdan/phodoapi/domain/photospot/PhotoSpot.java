package app.gangdan.phodoapi.domain.photospot;

import lombok.*;
import software.amazon.ion.Decimal;

import javax.persistence.*;

@Entity
@Table(name = "photo_spot")
@Getter
@Builder
@NoArgsConstructor @AllArgsConstructor
public class PhotoSpot {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long photoSpotId;

    private String photoSpotName;

    private Decimal longitude;

    private Decimal latitude;

}
